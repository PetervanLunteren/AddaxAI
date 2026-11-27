# BoT-SORT, trimmed to what the tracking script uses.
#
# Ported from the original implementation:
#   BoT-SORT, https://github.com/NirAharon/BoT-SORT
#   Copyright (c) 2022 Nir Aharon
# which builds on:
#   ByteTrack, https://github.com/ifzhang/ByteTrack
#   Copyright (c) 2021 Yifu Zhang
#
# Both are released under the MIT License:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""The tracker the tracking script runs: BoT-SORT without ReID.

Until 2026-11 the script ran ultralytics' BoT-SORT, which is AGPL-3.0. This
file replaces it with the MIT original, cut down to IoU association, the
xywh Kalman filter and sparse optical flow camera motion, and aligned with
ultralytics 8.3.177 wherever the two differed, so the swap changed no track:

1. A box at exactly the high threshold is a high box (``>=``, not ``>``).
2. No score fusion: the cost is IoU alone, as SharkTrack's tuning had it.
3. Input rows are ``x1, y1, x2, y2, conf, cls``; conf is not multiplied by cls.
4. IoU in float32 without the +1 pixel convention, eps 1e-7.
5. A box takes ultralytics' dtype path into the track (float32 centre and
   size, float64 corner, float32 stored), so every Kalman state matches.
6. Camera motion falls back to no motion when it fails or has no features,
   and is skipped when no frame is given.
7. ``update`` returns the detector's own row for each track, confirmed and
   just born separately, instead of the Kalman box.

Proved identical on 2026-11-09 (DEVELOPERS.md, "The tracker is ours, and MIT").

No ``app.*`` imports: it runs inside the model environments, loaded by path
from ``tracking_script.py``.
"""

import cv2
import lap
import numpy as np
import scipy.linalg

# A new track that is not matched on the next frame is dropped; a matched
# one still needs this IoU with its next box (ByteTrack's value).
UNCONFIRMED_MATCH_THRESH = 0.7
# The second association, of tracks left over against low-score boxes.
SECOND_MATCH_THRESH = 0.5
# Two tracks this close (IoU distance) are one; the younger goes.
DUPLICATE_THRESH = 0.15
# Removed tracks remembered, so a lost list never takes one back.
REMOVED_KEPT = 999


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class KalmanFilter:
    """Constant velocity model over (cx, cy, w, h) and their velocities."""

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = np.asarray([np.diag(sqr[i]) for i in range(len(mean))])
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance


def _tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    return ret


class Track:
    """One animal: a detection until activated, a Kalman state after.
    ``det_index`` is the row of this frame's detections the track holds."""

    def __init__(self, xyxy: np.ndarray, score: np.float32, cls: np.float32, det_index: int):
        # Centre and size in float32, then promoted to float64 (ultralytics
        # joins them with an index column), the corner taken there, and
        # stored as float32: the path that gives the same Kalman state.
        xywh = np.empty_like(xyxy)
        xywh[0] = (xyxy[0] + xyxy[2]) / 2
        xywh[1] = (xyxy[1] + xyxy[3]) / 2
        xywh[2] = xyxy[2] - xyxy[0]
        xywh[3] = xyxy[3] - xyxy[1]
        ltwh = xywh.astype(np.float64)
        ltwh[0] = ltwh[0] - ltwh[2] / 2
        ltwh[1] = ltwh[1] - ltwh[3] / 2
        self._tlwh = np.asarray(ltwh, dtype=np.float32)
        self.score = score
        self.cls = cls
        self.det_index = det_index
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.kalman_filter: KalmanFilter | None = None
        self.is_activated = False
        self.state = TrackState.New
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0

    @property
    def end_frame(self) -> int:
        return self.frame_id

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self) -> np.ndarray:
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def activate(self, kalman_filter: KalmanFilter, frame_id: int, track_id: int) -> None:
        self.kalman_filter = kalman_filter
        self.track_id = track_id
        self.mean, self.covariance = kalman_filter.initiate(_tlwh_to_xywh(self._tlwh))
        self.state = TrackState.Tracked
        # Only the first frame of a video confirms a track at once; later a
        # track is confirmed by its second matched box.
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def update(self, det: "Track", frame_id: int) -> None:
        """Take a matched box; ``re_activate`` is the same for a lost track."""
        self.frame_id = frame_id
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, _tlwh_to_xywh(det.tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = det.score
        self.cls = det.cls
        self.det_index = det.det_index

    re_activate = update

    @staticmethod
    def multi_predict(tracks: list["Track"], kalman_filter: KalmanFilter) -> None:
        if not tracks:
            return
        multi_mean = np.asarray([t.mean.copy() for t in tracks])
        multi_covariance = np.asarray([t.covariance for t in tracks])
        for i, t in enumerate(tracks):
            if t.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = kalman_filter.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance, strict=True)):
            tracks[i].mean = mean
            tracks[i].covariance = cov

    @staticmethod
    def multi_gmc(tracks: list["Track"], H: np.ndarray) -> None:
        if not tracks:
            return
        multi_mean = np.asarray([t.mean.copy() for t in tracks])
        multi_covariance = np.asarray([t.covariance for t in tracks])
        R8x8 = np.kron(np.eye(4, dtype=float), H[:2, :2])
        t = H[:2, 2]
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance, strict=True)):
            mean = R8x8.dot(mean)
            mean[:2] += t
            tracks[i].mean = mean
            tracks[i].covariance = R8x8.dot(cov).dot(R8x8.transpose())


def iou_cost(a: list[np.ndarray], b: list[np.ndarray]) -> np.ndarray:
    """1 - IoU between two lists of xyxy boxes, float32, no +1 pixel."""
    ious = np.zeros((len(a), len(b)), dtype=np.float32)
    if len(a) and len(b):
        a_x1, a_y1, a_x2, a_y2 = np.ascontiguousarray(a, dtype=np.float32).T
        b_x1, b_y1, b_x2, b_y2 = np.ascontiguousarray(b, dtype=np.float32).T
        inter = (np.minimum(a_x2[:, None], b_x2) - np.maximum(a_x1[:, None], b_x1)).clip(0) * (
            np.minimum(a_y2[:, None], b_y2) - np.maximum(a_y1[:, None], b_y1)
        ).clip(0)
        union = (b_x2 - b_x1) * (b_y2 - b_y1)
        union = union + ((a_x2 - a_x1) * (a_y2 - a_y1))[:, None] - inter
        ious = inter / (union + 1e-7)
    return 1 - ious


def _track_cost(a: list[Track], b: list[Track]) -> np.ndarray:
    return iou_cost([t.xyxy for t in a], [t.xyxy for t in b])


def linear_assignment(cost: np.ndarray, thresh: float) -> tuple[list, np.ndarray, np.ndarray]:
    """Matched (row, column) pairs and the unmatched rows and columns."""
    if cost.size == 0:
        return [], np.arange(cost.shape[0]), np.arange(cost.shape[1])
    _, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=thresh)
    matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
    return matches, np.where(x < 0)[0], np.where(y < 0)[0]


class GMC:
    """Camera motion between frames, by sparse optical flow on a frame
    shrunk by ``downscale``; the shift is scaled back to full frame."""

    def __init__(self, downscale: int):
        self.downscale = max(1, downscale)
        self.feature_params = dict(
            maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3,
            useHarrisDetector=False, k=0.04,
        )
        self.prev_frame: np.ndarray | None = None
        self.prev_keypoints: np.ndarray | None = None

    def apply(self, raw_frame: np.ndarray) -> np.ndarray:
        """The 2x3 affine transform from the previous frame to this one.
        Raises when RANSAC finds none; the caller then assumes no motion and
        this frame is not remembered, as in ultralytics."""
        height, width, channels = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY) if channels == 3 else raw_frame
        H = np.eye(2, 3)
        if self.downscale > 1:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)
        # First frame, or a previous frame without features (a black start
        # to a drop): nothing to follow yet.
        if self.prev_frame is None or self.prev_keypoints is None:
            self.prev_frame = frame.copy()
            self.prev_keypoints = None if keypoints is None else keypoints.copy()
            return H
        matched, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, frame, self.prev_keypoints, None
        )
        found = status.ravel().astype(bool)
        prev_points = self.prev_keypoints[found]
        curr_points = matched[found]
        if prev_points.shape[0] > 4:
            H, _ = cv2.estimateAffinePartial2D(prev_points, curr_points, cv2.RANSAC)
            if H is None:
                raise RuntimeError("no camera motion found")
            H[0, 2] *= self.downscale
            H[1, 2] *= self.downscale
        self.prev_frame = frame.copy()
        self.prev_keypoints = None if keypoints is None else keypoints.copy()
        return H


def _joint(a: list[Track], b: list[Track]) -> list[Track]:
    ids = {t.track_id for t in a}
    return a + [t for t in b if t.track_id not in ids]


def _sub(a: list[Track], b: list[Track]) -> list[Track]:
    ids = {t.track_id for t in b}
    return [t for t in a if t.track_id not in ids]


def _remove_duplicates(a: list[Track], b: list[Track]) -> tuple[list[Track], list[Track]]:
    """Of two tracks on the same animal, keep the one tracked longer."""
    pairs = np.where(_track_cost(a, b) < DUPLICATE_THRESH)
    dup_a, dup_b = set(), set()
    for p, q in zip(*pairs, strict=True):
        if a[p].frame_id - a[p].start_frame > b[q].frame_id - b[q].start_frame:
            dup_b.add(q)
        else:
            dup_a.add(p)
    return (
        [t for i, t in enumerate(a) if i not in dup_a],
        [t for i, t in enumerate(b) if i not in dup_b],
    )


class BoTSORT:
    """One video's tracker. A track starts from a box at ``high_thresh`` or
    above and is extended by boxes down to ``low_thresh``; a lost track is
    kept ``track_buffer`` frames."""

    def __init__(
        self, *, high_thresh: float, low_thresh: float, match_thresh: float,
        track_buffer: int, gmc_downscale: int,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = track_buffer
        self.kalman_filter = KalmanFilter()
        self.gmc = GMC(gmc_downscale)
        self.tracked: list[Track] = []
        self.lost: list[Track] = []
        self.removed: list[Track] = []
        self.frame_id = 0
        self._last_id = 0

    def _new_tracks(self, detections: np.ndarray, rows: np.ndarray) -> list[Track]:
        return [Track(detections[i, :4], detections[i, 4], detections[i, 5], int(i)) for i in rows]

    def update(
        self, detections: np.ndarray, frame: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Take one sampled frame's boxes (float32 rows ``x1, y1, x2, y2,
        conf, cls``) and the frame for camera motion, or None to skip it.
        Returns ``(confirmed, born)`` as float32 rows ``x1, y1, x2, y2,
        track_id, conf, cls``, each the detector's own box the track took
        on this frame: the tracks confirmed so far, and the tracks started
        on this frame that are not confirmed yet."""
        if detections.ndim != 2 or detections.shape[1] != 6:
            raise ValueError(f"detections must be (N, 6) rows, got shape {detections.shape}")
        self.frame_id += 1
        activated, refound, lost, removed = [], [], [], []
        conf = detections[:, 4]
        high = self._new_tracks(detections, np.flatnonzero(conf >= self.high_thresh))
        second = self._new_tracks(
            detections, np.flatnonzero((conf > self.low_thresh) & (conf < self.high_thresh))
        )
        unconfirmed = [t for t in self.tracked if not t.is_activated]
        confirmed = [t for t in self.tracked if t.is_activated]

        # First association: tracks and lost tracks against the high boxes.
        pool = _joint(confirmed, self.lost)
        Track.multi_predict(pool, self.kalman_filter)
        if frame is not None:
            try:
                warp = self.gmc.apply(frame)
            except Exception:
                warp = np.eye(2, 3)
            Track.multi_gmc(pool, warp)
            Track.multi_gmc(unconfirmed, warp)
        matches, u_track, u_detection = linear_assignment(
            _track_cost(pool, high), self.match_thresh
        )
        for i_track, i_det in matches:
            track = pool[i_track]
            if track.state == TrackState.Tracked:
                track.update(high[i_det], self.frame_id)
                activated.append(track)
            else:
                track.re_activate(high[i_det], self.frame_id)
                refound.append(track)

        # Second association: tracks left over against the low boxes.
        leftover = [pool[i] for i in u_track if pool[i].state == TrackState.Tracked]
        matches, u_track, _ = linear_assignment(
            _track_cost(leftover, second), SECOND_MATCH_THRESH
        )
        for i_track, i_det in matches:
            leftover[i_track].update(second[i_det], self.frame_id)
            activated.append(leftover[i_track])
        for i in u_track:
            track = leftover[i]
            if track.state != TrackState.Lost:
                track.state = TrackState.Lost
                lost.append(track)

        # Tracks born on the last frame against the high boxes still free;
        # one not matched now is dropped.
        high = [high[i] for i in u_detection]
        matches, u_unconfirmed, u_detection = linear_assignment(
            _track_cost(unconfirmed, high), UNCONFIRMED_MATCH_THRESH
        )
        for i_track, i_det in matches:
            unconfirmed[i_track].update(high[i_det], self.frame_id)
            activated.append(unconfirmed[i_track])
        for i in u_unconfirmed:
            unconfirmed[i].state = TrackState.Removed
            removed.append(unconfirmed[i])

        # Every high box still free starts a track (they are all at or above
        # the new-track threshold, which is the high threshold).
        for i in u_detection:
            self._last_id += 1
            high[i].activate(self.kalman_filter, self.frame_id, self._last_id)
            activated.append(high[i])

        for track in self.lost:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.state = TrackState.Removed
                removed.append(track)

        # The order below is ByteTrack's and is kept on purpose: ``lost`` is
        # cleaned against the tracks removed up to the previous frame, so a
        # lost track expired on this frame stays one frame longer and can
        # still be found again. Changing that changes which track a box joins.
        self.tracked = [t for t in self.tracked if t.state == TrackState.Tracked]
        self.tracked = _joint(self.tracked, activated)
        self.tracked = _joint(self.tracked, refound)
        self.lost = _sub(self.lost, self.tracked)
        self.lost.extend(lost)
        self.lost = _sub(self.lost, self.removed)
        self.tracked, self.lost = _remove_duplicates(self.tracked, self.lost)
        self.removed.extend(removed)
        if len(self.removed) > REMOVED_KEPT + 1:
            self.removed = self.removed[-REMOVED_KEPT:]

        return (
            self._rows(detections, [t for t in self.tracked if t.is_activated]),
            self._rows(detections, [t for t in self.tracked if not t.is_activated]),
        )

    @staticmethod
    def _rows(detections: np.ndarray, tracks: list[Track]) -> np.ndarray:
        rows = [
            [*detections[t.det_index, :4], t.track_id, t.score, t.cls] for t in tracks
        ]
        return np.asarray(rows, dtype=np.float32).reshape(-1, 7)

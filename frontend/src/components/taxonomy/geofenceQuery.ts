/** Whether a model selection should request geographic classification rules. */
export function shouldFetchModelGeofence(
  modelId: string | null | undefined,
  isDetectionAlias: boolean,
): boolean {
  return Boolean(modelId) && !isDetectionAlias;
}

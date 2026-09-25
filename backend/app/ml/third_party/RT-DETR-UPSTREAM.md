# RT-DETRv2 upstream source

The `rtdetrv2_pytorch` directory is copied from the official
[`lyuwenyu/RT-DETR`](https://github.com/lyuwenyu/RT-DETR) repository at commit
`29320b6fd828f8e0987a71426cf2d961b09dfed7` (2026-09-25). The upstream Apache
2.0 license is included as `RT-DETR-LICENSE`.

The application bundles this source for isolated RT-DETRv2 inference. User
configs are parsed with `yaml.safe_load`, include paths are constrained, and
the upstream config loader only receives a flattened file generated from the
validated config tree. No model pack can select Python imports or source code.

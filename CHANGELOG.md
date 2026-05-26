# Changelog

All notable changes to Pluvio are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Data layer rewritten against the real KMI app API.** Replaced the WMS-based
  `KmiRadarSource` + `KmiNowcastSource` pair with a single
  `KmiAppApiSource` hitting `app.meteo.be/services/appv4/?s=getForecasts`,
  which returns animation frames + per-location values in one signed call.
  Radar layer name was wrong (real: `belgian_rainfall_composite` w/ style
  `rainfall`); WMS endpoint is observation-only; nowcast comes from the
  unofficial app API used by the upstream Apache-2.0 `irm-kmi-api` package.
- Radar map now uses `OverlayImageLayer` (composite PNG over a geographic
  bbox) instead of `TileLayer` (slippy tile pyramid).
- Domain consolidated: `Nowcast`/`NowcastPoint` merged into
  `RadarAnimation`/`RadarFrame` (each frame carries its own per-location
  precipitation rate).
- Test fixtures replaced: `kmi_get_forecasts_sample.json` is a redacted real
  response capture.

### Added
- `KmiApiSigning` — daily-rotating md5 signing helper.
- 9 new tests covering signing, the DTO parser, and the rewritten source.

### Added in v0.1.0
- Initial project scaffold: Flutter 3.44, fvm-pinned.
- Layered radar feature: domain / data / application / presentation.
- KMI WMS radar source + per-location nowcast source.
- Brussels-centred map with animated radar timeline.
- Riverpod state, Dio HTTP with retry + error mapping, Talker logging.
- Localization in NL, FR, DE, EN.
- Test pyramid: domain, DTO, source, repository, widget tests.
- Fastlane lanes for Android (verify, build_aab, deploy_internal, deploy_production) and iOS (verify, build_ipa, deploy_testflight, deploy_production).
- GitHub Actions CI: format, analyze, test, build APK/iOS debug.

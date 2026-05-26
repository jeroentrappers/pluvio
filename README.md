# Pluvio

Open-source precipitation radar and 2-hour nowcast for Belgium and the Low Countries.

Pluvio is an alternative to paid radar apps, built on top of free data from the **Royal Meteorological Institute of Belgium** (KMI / IRM) and the **Royal Netherlands Meteorological Institute** (KNMI). It runs on Android and iOS, has no ads, no trackers, and no account.

## Features

- Animated radar timeline — past observations through 2-hour forecast.
- Per-location nowcast: minutes-until-rain, intensity, duration.
- Trilingual UI: **NL**, **FR**, **DE**, **EN**.
- Offline-friendly: tiles are cached; nowcasts refresh every 5 minutes.
- 100% open data: KMI open data (CC BY 4.0), KNMI Data Platform (CC BY 4.0).

## Project layout

```
lib/
  main.dart                          # zone-guarded entry point, logger wiring
  app/                               # MaterialApp, theme, root widget
  core/
    config/env.dart                  # dart-define backed runtime config
    logging/                         # Talker logger + Riverpod observer
    networking/                      # Dio factory, retry, error mapping
    result/                          # Result<T, E> sealed type
  features/
    location/                        # GPS + permission handling
    radar/
      domain/                        # pure Dart entities + repository contract
      data/                          # KMI sources, DTOs, repo implementation
      application/                   # Riverpod providers
      presentation/                  # screens & widgets
  l10n/                              # ARB files + generated AppLocalizations
test/                                # unit + widget tests, mirrors lib/
integration_test/                    # E2E flows under the integration_test binding
android/fastlane/                    # Android delivery lanes + Play metadata
ios/fastlane/                        # iOS delivery lanes + App Store metadata
.github/workflows/                   # CI: format check, analyze, test, builds
```

## Requirements

- [fvm](https://fvm.app) (Flutter Version Management). The pinned version lives in `.fvmrc`.
- Ruby 3.x + bundler (for Fastlane).
- For Android releases: Java 17, the Android SDK, and the signing material described in *Releasing → Android* below.
- For iOS releases: macOS, Xcode, a paid Apple Developer account, and a `match` git repository.

## Getting started

```bash
fvm install                        # installs the pinned Flutter SDK
fvm flutter pub get                # resolve dependencies
fvm flutter gen-l10n               # generate AppLocalizations
fvm flutter run                    # launch on a connected device
```

## Configuration

All runtime knobs come in through `--dart-define` — there is **never** a secret in code. The supported keys (defaults in parentheses):

| Key | Default | Purpose |
|---|---|---|
| `PLUVIO_KMI_APP_API_BASE_URL` | `https://app.meteo.be/services/appv4/` | Unofficial KMI mobile-app API used to fetch the animated radar + per-location nowcast in one call. |
| `PLUVIO_RADAR_BOUNDS_WEST`  | `1.5`  | Geographic west bound of the radar PNG, in decimal degrees. |
| `PLUVIO_RADAR_BOUNDS_EAST`  | `7.5`  | East bound. |
| `PLUVIO_RADAR_BOUNDS_SOUTH` | `48.9` | South bound. |
| `PLUVIO_RADAR_BOUNDS_NORTH` | `52.5` | North bound. |
| `PLUVIO_SENTRY_DSN` | _(empty)_ | Optional crash reporting. |

### Why the unofficial app API

KMI exposes radar and forecast data through two surfaces:

1. **`opendata.meteo.be`** (official, CC BY 4.0) — exposes a WMS at `/service/radar/wms` with layer `belgian_rainfall_composite` (styles: `rainfall`). Slippy-tile capable, but **observation-only** — no nowcast forecast. Confirmed by GetCapabilities: `Dimension name="time"` runs ~13 hours back and stops ~10 minutes before now.
2. **`app.meteo.be/services/appv4/`** (unofficial, the same endpoint KMI's own mobile app uses) — returns animation frames *and* per-location precipitation values for both the past hour and the **2-hour forecast** in a single signed `getForecasts` call. This is what powers Pluvio. The signing recipe is `md5("r9EnW374jkJ9acc;<method>;DD/MM/YYYY")`, lifted from the Apache-2.0 [`irm-kmi-api`](https://github.com/jdejaegh/irm-kmi-api) Python package that pioneered access to this API.

The opendata WMS remains a candidate for a future "high-zoom slippy radar" mode, since the app API ships a single 640×490 composite PNG per timestep rather than tile pyramids.

## Testing strategy

| Layer | Where | How |
|---|---|---|
| Pure domain logic (intensity buckets, nowcast helpers, animation indexing) | `test/features/*/domain/` | Plain `test()` cases, no mocks |
| DTO parsing | `test/features/*/data/models/` | Fixture JSON / XML under `test/_fixtures/` |
| HTTP sources | `test/features/*/data/sources/` | Mocked `Dio` via `mocktail`; all four DioException types covered |
| Repositories | `test/features/*/data/` | Sources stubbed, asserts the wire→domain mapping |
| Widgets that don't render maps | `test/features/*/presentation/widgets/` | Standard `pumpWidget` |
| Full screen flows | `integration_test/` | Run under `IntegrationTestWidgetsFlutterBinding` so map tile fetches work |

```bash
fvm flutter test --coverage         # unit + widget
fvm flutter test integration_test/  # E2E on a real device or emulator
```

## Releasing

### Android

1. Create an upload keystore. Store the path + credentials in either
   `android/key.properties` (gitignored) or the `FASTLANE_PLUVIO_*` env vars.
2. Place the Play Console JSON service account key somewhere outside the repo
   and set `FASTLANE_PLUVIO_PLAY_JSON_KEY` to its path.
3. Lanes:

```bash
bundle install
bundle exec fastlane android verify             # analyze + test
bundle exec fastlane android build_aab          # signed AAB
bundle exec fastlane android deploy_internal    # → Play internal track
bundle exec fastlane android deploy_production  # promote internal → production
```

### iOS

Set up `match` in App Store Connect with your certificates repo, then:

```bash
export FASTLANE_PLUVIO_MATCH_GIT_URL=git@github.com:appmire/pluvio-certs.git
export MATCH_PASSWORD=…
export FASTLANE_PLUVIO_APPLE_ID=…
export FASTLANE_PLUVIO_TEAM_ID=…
export FASTLANE_PLUVIO_ITC_TEAM_ID=…

bundle exec fastlane ios verify
bundle exec fastlane ios build_ipa
bundle exec fastlane ios deploy_testflight
bundle exec fastlane ios deploy_production
```

## Data sources & attribution

- **KMI / IRM (Belgium)** — radar composites + 2-hour nowcast from the official KMI mobile-app endpoint (`app.meteo.be/services/appv4`). The endpoint is unofficial and reverse-engineered for the open-source `irm-kmi-api` Python package; Pluvio uses the same signing recipe and follows the same conventions. The app surfaces "Radar © KMI / IRM" on every frame.
- **KNMI (Netherlands)** — planned for the cross-border extension; would require a free API key from the [KNMI Developer Portal](https://developer.dataplatform.knmi.nl/).
- **OpenStreetMap** — base map tiles. Attributed in-app.

### Endpoint risk

The KMI app API is **unofficial**. If KMI changes the signing salt, query parameter shape, or response schema, `KmiAppApiSource` and `KmiGetForecastsDto` will need to be updated to match. The DTO parser tolerates missing/extra fields, but a salt rotation would break authentication entirely. Mitigation: keep `[irm-kmi-api](https://github.com/jdejaegh/irm-kmi-api)` on a watchlist — the upstream maintainer ships fixes promptly when the API drifts.

## Licence

Released under the **GNU General Public License v3.0**. See [LICENSE](./LICENSE). Pluvio is FOSS in spirit and in licence: derivative works must remain open under the same terms.

## Status

Foundational scaffold, data layer validated against live KMI endpoints (see commit log). Not yet on the stores. The radar PNG bounds (`PLUVIO_RADAR_BOUNDS_*`) are best-effort defaults — calibrate them against the rendered overlay on a real device before first release.

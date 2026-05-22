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
| `PLUVIO_KMI_BASE_URL` | `https://app.meteo.be/services/appviewer` | KMI per-location nowcast |
| `PLUVIO_KMI_OPENDATA_BASE_URL` | `https://opendata.meteo.be/service` | Root of the KMI open-data services |
| `PLUVIO_KMI_RADAR_WMS_URL` | `https://opendata.meteo.be/service/radar/wms` | WMS endpoint serving radar tiles |
| `PLUVIO_KMI_RADAR_LAYER` | `RADAR.BE_COMPOSITE` | Layer name within the WMS service |
| `PLUVIO_KNMI_API_KEY` | _(empty)_ | Required only if KNMI Dutch nowcast is enabled |
| `PLUVIO_SENTRY_DSN` | _(empty)_ | Optional crash reporting |

Verify the KMI endpoints against the canonical docs at <https://opendata.meteo.be> before the first release.

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

- **KMI / IRM (Belgium)** — radar composites and nowcast via `opendata.meteo.be`. Licensed CC BY 4.0; the app surfaces a "Radar © KMI / IRM" credit on every map tile.
- **KNMI (Netherlands)** — used for the cross-border nowcast extension. Requires an API key from the [KNMI Developer Portal](https://developer.dataplatform.knmi.nl/). Licensed CC BY 4.0.
- **OpenStreetMap** — base map tiles. Attributed in-app.

## Licence

Released under the **GNU General Public License v3.0**. See [LICENSE](./LICENSE). Pluvio is FOSS in spirit and in licence: derivative works must remain open under the same terms.

## Status

Foundational scaffold — not yet on the stores. Endpoints in `core/config/env.dart` need to be validated against current KMI documentation before the first public build.

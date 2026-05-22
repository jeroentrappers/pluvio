/// Runtime configuration sourced from `--dart-define` flags.
///
/// Secrets and tunables never live in code. The CI workflows / fastlane lanes
/// inject these at build time; see `dart_defines/` and the README for the full
/// set of supported keys.
abstract final class Env {
  static const String kmiBaseUrl = String.fromEnvironment(
    'PLUVIO_KMI_BASE_URL',
    defaultValue: 'https://app.meteo.be/services/appviewer',
  );

  static const String kmiOpenDataBaseUrl = String.fromEnvironment(
    'PLUVIO_KMI_OPENDATA_BASE_URL',
    defaultValue: 'https://opendata.meteo.be/service',
  );

  static const String kmiRadarWmsUrl = String.fromEnvironment(
    'PLUVIO_KMI_RADAR_WMS_URL',
    defaultValue: 'https://opendata.meteo.be/service/radar/wms',
  );

  static const String kmiRadarLayer = String.fromEnvironment(
    'PLUVIO_KMI_RADAR_LAYER',
    defaultValue: 'RADAR.BE_COMPOSITE',
  );

  /// Optional KNMI API key — only required if we fetch Dutch nowcasts directly.
  static const String knmiApiKey = String.fromEnvironment('PLUVIO_KNMI_API_KEY');

  static const String sentryDsn = String.fromEnvironment('PLUVIO_SENTRY_DSN');

  static const bool isProduction = bool.fromEnvironment('dart.vm.product');

  /// Fail fast at startup if a required setting is missing/empty.
  static void assertValid() {
    assert(kmiBaseUrl.isNotEmpty, 'PLUVIO_KMI_BASE_URL must not be empty');
    assert(kmiRadarWmsUrl.isNotEmpty, 'PLUVIO_KMI_RADAR_WMS_URL must not be empty');
  }
}

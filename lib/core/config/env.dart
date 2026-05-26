/// Runtime configuration sourced from `--dart-define` flags.
///
/// Secrets and tunables never live in code. The CI workflows / fastlane lanes
/// inject these at build time; see the README for the full set of supported keys.
abstract final class Env {
  /// Unofficial KMI mobile-app API. Used by the upstream Apache-2.0
  /// `irm-kmi-api` package; gives per-location nowcast values + 2-hour
  /// radar animation frames in one call.
  static const String kmiAppApiBaseUrl = String.fromEnvironment(
    'PLUVIO_KMI_APP_API_BASE_URL',
    defaultValue: 'https://app.meteo.be/services/appv4/',
  );

  /// Geographic bounds the KMI radar composite PNG is rendered onto. The
  /// official extent isn't published; these values match the KMI app's
  /// "Belgian rainfall composite" framing and can be tuned via dart-define.
  /// Doubles aren't valid for `fromEnvironment`; we pass strings and parse.
  static double get radarBoundsWest => _parseDouble('PLUVIO_RADAR_BOUNDS_WEST', 1.5);
  static double get radarBoundsEast => _parseDouble('PLUVIO_RADAR_BOUNDS_EAST', 7.5);
  static double get radarBoundsSouth => _parseDouble('PLUVIO_RADAR_BOUNDS_SOUTH', 48.9);
  static double get radarBoundsNorth => _parseDouble('PLUVIO_RADAR_BOUNDS_NORTH', 52.5);

  static double _parseDouble(String key, double fallback) {
    final raw = String.fromEnvironment(key);
    if (raw.isEmpty) return fallback;
    return double.tryParse(raw) ?? fallback;
  }

  /// Optional crash-reporting DSN.
  static const String sentryDsn = String.fromEnvironment('PLUVIO_SENTRY_DSN');

  static const bool isProduction = bool.fromEnvironment('dart.vm.product');

  static void assertValid() {
    assert(kmiAppApiBaseUrl.isNotEmpty, 'PLUVIO_KMI_APP_API_BASE_URL must not be empty');
    assert(radarBoundsEast > radarBoundsWest, 'radar bounds: east must be > west');
    assert(radarBoundsNorth > radarBoundsSouth,
        'radar bounds: north must be > south');
  }
}

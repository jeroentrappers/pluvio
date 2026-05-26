import 'package:geolocator/geolocator.dart';
import 'package:latlong2/latlong.dart';

import '../../../core/logging/logger.dart';

/// Wraps `geolocator` so we can test consumers without hitting platform APIs.
abstract interface class LocationService {
  /// Returns the device's current position, or `null` when it's unavailable
  /// (location services off, permission denied, timeout, misconfigured
  /// platform manifest, etc.). The fallback to a default location is the
  /// caller's responsibility — we never throw here so the UI always renders.
  Future<LatLng?> currentLocation();
}

class GeolocatorLocationService implements LocationService {
  const GeolocatorLocationService();

  /// Brussels city centre — used as the default when no real fix is available.
  static const LatLng fallback = LatLng(50.8503, 4.3517);

  @override
  Future<LatLng?> currentLocation() async {
    try {
      final enabled = await Geolocator.isLocationServiceEnabled();
      if (!enabled) {
        AppLogger.talker.info('Location services disabled; using fallback.');
        return null;
      }

      var permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }
      if (permission == LocationPermission.denied ||
          permission == LocationPermission.deniedForever) {
        AppLogger.talker
            .info('Location permission $permission; using fallback.');
        return null;
      }

      final pos = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.medium,
          timeLimit: Duration(seconds: 10),
        ),
      );
      return LatLng(pos.latitude, pos.longitude);
    } on Object catch (e, st) {
      // Manifest missing, GPS timeout, plugin not registered on web/desktop,
      // … logged and swallowed so the UI can fall back to Brussels instead
      // of showing an error screen. The user can override location later.
      AppLogger.captureException(e, st, 'GeolocatorLocationService');
      return null;
    }
  }
}

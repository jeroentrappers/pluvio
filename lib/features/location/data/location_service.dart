import 'package:geolocator/geolocator.dart';
import 'package:latlong2/latlong.dart';

/// Wraps `geolocator` so we can test consumers without hitting platform APIs.
abstract interface class LocationService {
  Future<LatLng?> currentLocation();
}

class GeolocatorLocationService implements LocationService {
  const GeolocatorLocationService();

  /// Brussels city centre — used as a sensible default when the device hasn't
  /// granted permission yet, so the map always opens on Belgium.
  static const LatLng fallback = LatLng(50.8503, 4.3517);

  @override
  Future<LatLng?> currentLocation() async {
    final enabled = await Geolocator.isLocationServiceEnabled();
    if (!enabled) return null;

    var permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
    }
    if (permission == LocationPermission.denied ||
        permission == LocationPermission.deniedForever) {
      return null;
    }

    final pos = await Geolocator.getCurrentPosition(
      locationSettings: const LocationSettings(
        accuracy: LocationAccuracy.medium,
        timeLimit: Duration(seconds: 10),
      ),
    );
    return LatLng(pos.latitude, pos.longitude);
  }
}

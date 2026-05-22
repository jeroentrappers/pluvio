import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:latlong2/latlong.dart';

import '../data/location_service.dart';

final locationServiceProvider = Provider<LocationService>((ref) {
  return const GeolocatorLocationService();
});

/// Latest known device location, or [GeolocatorLocationService.fallback] if
/// permissions/services aren't available. Always emits something so the UI
/// never has to special-case "no location yet".
final currentLocationProvider = FutureProvider<LatLng>((ref) async {
  final svc = ref.watch(locationServiceProvider);
  final loc = await svc.currentLocation();
  return loc ?? GeolocatorLocationService.fallback;
});

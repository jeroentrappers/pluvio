import 'package:latlong2/latlong.dart';

import '../../../core/networking/api_failure.dart';
import '../../../core/result/result.dart';
import 'radar_animation.dart';

/// The contract every radar data source must satisfy. Letting providers
/// depend on this interface instead of a concrete class is what makes the
/// presentation layer testable without HTTP.
abstract interface class RadarRepository {
  /// Fetches the full radar animation for [location]. Each frame carries its
  /// own per-location precipitation rate so the caller doesn't need a
  /// separate nowcast call.
  Future<Result<RadarAnimation, ApiFailure>> fetchAnimation(LatLng location);
}

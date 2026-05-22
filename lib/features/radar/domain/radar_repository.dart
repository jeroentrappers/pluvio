import 'package:latlong2/latlong.dart';

import '../../../core/networking/api_failure.dart';
import '../../../core/result/result.dart';
import 'nowcast.dart';
import 'radar_animation.dart';

/// The contract every radar data source must satisfy. Letting providers depend
/// on this interface instead of a concrete class is what makes the
/// presentation layer testable without HTTP.
abstract interface class RadarRepository {
  Future<Result<Nowcast, ApiFailure>> fetchNowcast(LatLng location);

  Future<Result<RadarAnimation, ApiFailure>> fetchRadarAnimation();
}

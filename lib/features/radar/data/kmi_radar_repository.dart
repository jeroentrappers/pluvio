import 'package:latlong2/latlong.dart';

import '../../../core/networking/api_failure.dart';
import '../../../core/result/result.dart';
import '../domain/radar_animation.dart';
import '../domain/radar_repository.dart';
import 'sources/kmi_app_api_source.dart';

/// Production radar repository. One call to the KMI mobile-app endpoint
/// returns both the animation frames *and* the per-location precipitation
/// rate per frame, so we don't need a separate nowcast call.
class KmiRadarRepository implements RadarRepository {
  KmiRadarRepository({
    required this.source,
    DateTime Function()? clock,
  }) : _clock = clock ?? DateTime.now;

  final KmiAppApiSource source;
  final DateTime Function() _clock;

  /// Converts the wire-format mm/10min into the mm/h used throughout the
  /// domain so [PrecipitationLevel] thresholds stay in their canonical units.
  static const _mmPer10MinToMmPerHour = 6.0;

  @override
  Future<Result<RadarAnimation, ApiFailure>> fetchAnimation(LatLng location) async {
    final res = await source.fetch(
      latitude: location.latitude,
      longitude: location.longitude,
    );

    return res.when(
      ok: (dto) {
        final frames = [
          for (final f in dto.animation.sequence)
            RadarFrame(
              timestamp: f.time,
              imageUrl: f.imageUrl,
              valueMmPerHour: f.valueMmPer10Min * _mmPer10MinToMmPerHour,
            ),
        ]..sort((a, b) => a.timestamp.compareTo(b.timestamp));

        return Result.ok(
          RadarAnimation(
            frames: frames,
            referenceTime: _clock().toUtc(),
            location: location,
          ),
        );
      },
      err: Result.err,
    );
  }
}

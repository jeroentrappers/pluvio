import 'package:latlong2/latlong.dart';

import '../../../core/networking/api_failure.dart';
import '../../../core/result/result.dart';
import '../domain/radar_animation.dart';
import '../domain/radar_repository.dart';
import 'sources/pluvio_backend_source.dart';

/// Repository backed by the Pluvio backend. Maps the backend's `/v1/forecast`
/// response onto the domain [RadarAnimation]. Keeps every band the backend
/// returns (nowcast / short / medium / long) so the UI can render whatever
/// horizon is currently cached.
class PluvioBackendRepository implements RadarRepository {
  PluvioBackendRepository({
    required this.source,
    DateTime Function()? clock,
  }) : _clock = clock ?? DateTime.now;

  final PluvioBackendSource source;
  final DateTime Function() _clock;

  @override
  Future<Result<RadarAnimation, ApiFailure>> fetchAnimation(LatLng location) async {
    final res = await source.fetchForecast(
      latitude: location.latitude,
      longitude: location.longitude,
    );

    return res.when(
      ok: (dto) {
        final frames = [
          for (final f in dto.frames)
            RadarFrame(
              timestamp: f.validTime,
              imageUrl: source.absoluteUrl(f.overlayUrl),
              valueMmPerHour: f.rateMmPerHour,
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

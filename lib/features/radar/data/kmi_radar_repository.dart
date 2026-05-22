import 'package:latlong2/latlong.dart';

import '../../../core/networking/api_failure.dart';
import '../../../core/result/result.dart';
import '../domain/nowcast.dart';
import '../domain/radar_animation.dart';
import '../domain/radar_repository.dart';
import 'sources/kmi_nowcast_source.dart';
import 'sources/kmi_radar_source.dart';

/// Production radar repository: orchestrates the two KMI data sources and
/// maps wire-format DTOs to domain models.
class KmiRadarRepository implements RadarRepository {
  KmiRadarRepository({
    required this.nowcastSource,
    required this.radarSource,
    DateTime Function()? clock,
  }) : _clock = clock ?? DateTime.now;

  final KmiNowcastSource nowcastSource;
  final KmiRadarSource radarSource;
  final DateTime Function() _clock;

  @override
  Future<Result<Nowcast, ApiFailure>> fetchNowcast(LatLng location) async {
    final res = await nowcastSource.fetch(
      latitude: location.latitude,
      longitude: location.longitude,
    );

    return res.when(
      ok: (dto) {
        final step = Duration(minutes: dto.intervalMinutes);
        final points = <NowcastPoint>[];
        for (var i = 0; i < dto.precipitationMmPerHour.length; i++) {
          points.add(NowcastPoint(
            timestamp: dto.issuedAt.add(step * i),
            precipitationMmPerHour: dto.precipitationMmPerHour[i],
          ));
        }
        return Result.ok(
          Nowcast(location: location, issuedAt: dto.issuedAt, points: points),
        );
      },
      err: Result.err,
    );
  }

  @override
  Future<Result<RadarAnimation, ApiFailure>> fetchRadarAnimation() async {
    final res = await radarSource.fetchCapabilities();
    return res.when(
      ok: (caps) {
        final frames = caps.timeSteps
            .map((t) => RadarFrame(
                  timestamp: t,
                  tileUrlTemplate: radarSource.tileTemplateForFrame(t).toString(),
                ))
            .toList(growable: false);
        return Result.ok(
          RadarAnimation(frames: frames, referenceTime: _clock().toUtc()),
        );
      },
      err: Result.err,
    );
  }
}

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:latlong2/latlong.dart';

import '../../../core/config/env.dart';
import '../../../core/networking/api_failure.dart';
import '../../../core/networking/dio_client.dart';
import '../../../core/result/result.dart';
import '../data/kmi_radar_repository.dart';
import '../data/sources/kmi_app_api_source.dart';
import '../domain/radar_animation.dart';
import '../domain/radar_repository.dart';

final radarRepositoryProvider = Provider<RadarRepository>((ref) {
  final dio = ref.watch(dioProvider);
  return KmiRadarRepository(
    source: KmiAppApiSource(dio: dio, baseUrl: Env.kmiAppApiBaseUrl),
  );
});

/// Auto-refreshing animation for a given location. The KMI mobile API
/// recomputes a new frame every 10 minutes — re-poll on that cadence so the
/// timeline never drifts more than half a frame stale.
final radarAnimationProvider =
    FutureProvider.autoDispose.family<Result<RadarAnimation, ApiFailure>, LatLng>(
  (ref, location) async {
    final repo = ref.watch(radarRepositoryProvider);
    final timer = Stream<void>.periodic(const Duration(minutes: 5));
    final sub = timer.listen((_) => ref.invalidateSelf());
    ref.onDispose(sub.cancel);
    return repo.fetchAnimation(location);
  },
);

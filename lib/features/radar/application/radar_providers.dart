import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:latlong2/latlong.dart';

import '../../../core/config/env.dart';
import '../../../core/networking/api_failure.dart';
import '../../../core/networking/dio_client.dart';
import '../../../core/result/result.dart';
import '../data/pluvio_backend_repository.dart';
import '../data/sources/pluvio_backend_source.dart';
import '../domain/radar_animation.dart';
import '../domain/radar_repository.dart';

final radarRepositoryProvider = Provider<RadarRepository>((ref) {
  final dio = ref.watch(dioProvider);
  return PluvioBackendRepository(
    source: PluvioBackendSource(dio: dio, baseUrl: Env.pluvioApiBaseUrl),
  );
});

/// Auto-refreshing animation for a given location. The backend nowcast band
/// refreshes every 5 minutes — re-poll on that cadence so the timeline never
/// drifts more than one refresh stale.
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

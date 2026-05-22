import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:latlong2/latlong.dart';

import '../../../core/config/env.dart';
import '../../../core/networking/api_failure.dart';
import '../../../core/networking/dio_client.dart';
import '../../../core/result/result.dart';
import '../data/kmi_radar_repository.dart';
import '../data/sources/kmi_nowcast_source.dart';
import '../data/sources/kmi_radar_source.dart';
import '../domain/nowcast.dart';
import '../domain/radar_animation.dart';
import '../domain/radar_repository.dart';

final radarRepositoryProvider = Provider<RadarRepository>((ref) {
  final dio = ref.watch(dioProvider);
  return KmiRadarRepository(
    nowcastSource: KmiNowcastSource(dio: dio, baseUrl: Env.kmiBaseUrl),
    radarSource: KmiRadarSource(
      dio: dio,
      wmsUrl: Env.kmiRadarWmsUrl,
      layer: Env.kmiRadarLayer,
    ),
  );
});

/// Auto-refreshing radar animation. Refetches every 5 minutes since the KMI
/// composite is updated at that cadence.
final radarAnimationProvider =
    FutureProvider.autoDispose<Result<RadarAnimation, ApiFailure>>((ref) async {
  final repo = ref.watch(radarRepositoryProvider);
  final timer = Stream<void>.periodic(const Duration(minutes: 5));
  final sub = timer.listen((_) => ref.invalidateSelf());
  ref.onDispose(sub.cancel);
  return repo.fetchRadarAnimation();
});

/// Nowcast for a specific location. Parameterised so the same provider serves
/// the user's GPS-derived location and any other pinned spots later on.
final nowcastProvider =
    FutureProvider.autoDispose.family<Result<Nowcast, ApiFailure>, LatLng>(
  (ref, location) async {
    final repo = ref.watch(radarRepositoryProvider);
    return repo.fetchNowcast(location);
  },
);

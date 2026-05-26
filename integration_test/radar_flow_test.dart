import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hooks_riverpod/hooks_riverpod.dart';
import 'package:integration_test/integration_test.dart';
import 'package:latlong2/latlong.dart';
import 'package:pluvio/core/networking/api_failure.dart';
import 'package:pluvio/core/result/result.dart';
import 'package:pluvio/features/location/application/location_providers.dart';
import 'package:pluvio/features/location/data/location_service.dart';
import 'package:pluvio/features/radar/application/radar_providers.dart';
import 'package:pluvio/features/radar/domain/radar_animation.dart';
import 'package:pluvio/features/radar/domain/radar_repository.dart';
import 'package:pluvio/features/radar/presentation/radar_screen.dart';
import 'package:pluvio/l10n/app_localizations.dart';

class _FakeLoc implements LocationService {
  @override
  Future<LatLng?> currentLocation() async => const LatLng(50.85, 4.35);
}

class _Stub implements RadarRepository {
  _Stub(this.animation);

  final RadarAnimation animation;

  @override
  Future<Result<RadarAnimation, ApiFailure>> fetchAnimation(LatLng location) async =>
      Result.ok(animation);
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('Pluvio renders the radar screen end-to-end with stubbed data',
      (tester) async {
    final ref = DateTime.utc(2026, 5, 26, 8);
    final animation = RadarAnimation(
      frames: [
        RadarFrame(
          timestamp: ref,
          imageUrl: 'https://example.test/0.png',
          valueMmPerHour: 0,
        ),
      ],
      referenceTime: ref,
      location: const LatLng(50.85, 4.35),
    );

    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          locationServiceProvider.overrideWithValue(_FakeLoc()),
          radarRepositoryProvider.overrideWithValue(_Stub(animation)),
        ],
        child: MaterialApp(
          localizationsDelegates: AppLocalizations.localizationsDelegates,
          supportedLocales: AppLocalizations.supportedLocales,
          home: const RadarScreen(),
        ),
      ),
    );

    await tester.pumpAndSettle(const Duration(seconds: 1));

    expect(find.text('Pluvio'), findsWidgets);
    expect(find.byType(RadarScreen), findsOneWidget);
  });
}

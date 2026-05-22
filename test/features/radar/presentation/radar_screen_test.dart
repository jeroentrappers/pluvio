import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hooks_riverpod/hooks_riverpod.dart';
import 'package:latlong2/latlong.dart';
import 'package:pluvio/core/networking/api_failure.dart';
import 'package:pluvio/core/result/result.dart';
import 'package:pluvio/features/location/application/location_providers.dart';
import 'package:pluvio/features/location/data/location_service.dart';
import 'package:pluvio/features/radar/application/radar_providers.dart';
import 'package:pluvio/features/radar/domain/nowcast.dart';
import 'package:pluvio/features/radar/domain/radar_animation.dart';
import 'package:pluvio/features/radar/domain/radar_repository.dart';
import 'package:pluvio/features/radar/presentation/radar_screen.dart';
import 'package:pluvio/l10n/app_localizations.dart';

class _StubRepository implements RadarRepository {
  _StubRepository({required this.nowcast, required this.animation});

  final Nowcast nowcast;
  final RadarAnimation animation;

  @override
  Future<Result<Nowcast, ApiFailure>> fetchNowcast(LatLng location) async {
    return Result.ok(nowcast);
  }

  @override
  Future<Result<RadarAnimation, ApiFailure>> fetchRadarAnimation() async {
    return Result.ok(animation);
  }
}

class _FakeLocationService implements LocationService {
  @override
  Future<LatLng?> currentLocation() async => const LatLng(50.85, 4.35);
}

void main() {
  final issued = DateTime.utc(2026, 5, 22, 10);
  final nowcast = Nowcast(
    location: const LatLng(50.85, 4.35),
    issuedAt: issued,
    points: [
      NowcastPoint(timestamp: issued, precipitationMmPerHour: 0),
      NowcastPoint(
        timestamp: issued.add(const Duration(minutes: 5)),
        precipitationMmPerHour: 0,
      ),
      NowcastPoint(
        timestamp: issued.add(const Duration(minutes: 10)),
        precipitationMmPerHour: 1.5,
      ),
    ],
  );
  final animation = RadarAnimation(
    frames: [
      RadarFrame(
        timestamp: issued.subtract(const Duration(minutes: 5)),
        tileUrlTemplate: 'https://example.test/t1.png',
      ),
      RadarFrame(timestamp: issued, tileUrlTemplate: 'https://example.test/t2.png'),
    ],
    referenceTime: issued,
  );

  // Pinned to the IntegrationTest binding — flutter_map tile fetches never
  // settle under TestWidgetsFlutterBinding. See integration_test/radar_flow_test.dart.
  testWidgets(
    'renders the headline derived from the nowcast',
    skip: true,
    (tester) async {
    // The map widgets attempt network tile fetches that hang in the test
    // binding; HttpOverrides shortcuts them so the test can settle.
    await HttpOverrides.runZoned(
      () async {
        await tester.pumpWidget(
          ProviderScope(
            overrides: [
              locationServiceProvider.overrideWithValue(_FakeLocationService()),
              radarRepositoryProvider.overrideWithValue(
                _StubRepository(nowcast: nowcast, animation: animation),
              ),
            ],
            child: MaterialApp(
              localizationsDelegates: AppLocalizations.localizationsDelegates,
              supportedLocales: AppLocalizations.supportedLocales,
              locale: const Locale('en'),
              home: const RadarScreen(),
            ),
          ),
        );

        // Flush the location future + the nowcast/animation futures.
        for (var i = 0; i < 4; i++) {
          await tester.pump(const Duration(milliseconds: 50));
        }

        expect(find.text('Rain expected in 10 min.'), findsOneWidget);
        expect(find.byType(Slider), findsOneWidget);
      },
      createHttpClient: (_) => _NoopHttpClient(),
    );
  });
}

class _NoopHttpClient implements HttpClient {
  @override
  dynamic noSuchMethod(Invocation invocation) => null;
}

import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:latlong2/latlong.dart';
import 'package:mocktail/mocktail.dart';
import 'package:pluvio/core/networking/api_failure.dart';
import 'package:pluvio/features/radar/data/pluvio_backend_repository.dart';
import 'package:pluvio/features/radar/data/sources/pluvio_backend_source.dart';

class _MockDio extends Mock implements Dio {}

Map<String, dynamic> _sampleResponse() => {
      'issued_at': '2026-05-27T07:00:00Z',
      'location': {'lat': 50.85, 'lon': 4.35},
      'model_version': 'stub-0.1',
      'horizon_min': 120,
      'frames': [
        {
          'band': 'nowcast',
          'lead_min': 0,
          'valid_time': '2026-05-27T07:00:00Z',
          'rate_mm_per_h': 0.0,
          'overlay_url': '/v1/overlay/nowcast/0.png?t=2026-05-27T07-00-00Z',
        },
        {
          'band': 'nowcast',
          'lead_min': 10,
          'valid_time': '2026-05-27T07:10:00Z',
          'rate_mm_per_h': 1.5,
          'overlay_url': '/v1/overlay/nowcast/10.png?t=2026-05-27T07-00-00Z',
        },
        {
          // Non-nowcast band — also kept, so the UI can render the extended
          // horizon when the short/medium/long caches start populating.
          'band': 'short',
          'lead_min': 180,
          'valid_time': '2026-05-27T10:00:00Z',
          'rate_mm_per_h': 0.2,
          'overlay_url': '/v1/overlay/short/180.png?t=2026-05-27T07-00-00Z',
        },
      ],
    };

void main() {
  late _MockDio dio;
  late PluvioBackendRepository repo;

  setUp(() {
    dio = _MockDio();
    repo = PluvioBackendRepository(
      source: PluvioBackendSource(dio: dio, baseUrl: 'https://pluvio.appmire.be'),
      clock: () => DateTime.utc(2026, 5, 27, 7),
    );
  });

  test('maps every band frame to RadarAnimation with absolute overlay URLs', () async {
    when(() => dio.get<Map<String, dynamic>>(
          any(),
          queryParameters: any(named: 'queryParameters'),
        )).thenAnswer((_) async => Response<Map<String, dynamic>>(
          requestOptions: RequestOptions(path: ''),
          data: _sampleResponse(),
          statusCode: 200,
        ));

    final res = await repo.fetchAnimation(const LatLng(50.85, 4.35));

    expect(res.isOk, isTrue);
    final anim = res.valueOrNull!;
    // Both nowcast frames AND the short-band frame survive — we keep the
    // full horizon so the UI can render whatever's cached.
    expect(anim.frames.length, 3);
    expect(anim.location, const LatLng(50.85, 4.35));
    expect(
      anim.frames.first.imageUrl,
      'https://pluvio.appmire.be/v1/overlay/nowcast/0.png?t=2026-05-27T07-00-00Z',
    );
    expect(anim.frames[1].valueMmPerHour, 1.5);
    expect(
      anim.frames.last.imageUrl,
      'https://pluvio.appmire.be/v1/overlay/short/180.png?t=2026-05-27T07-00-00Z',
    );
    expect(anim.minutesUntilRain, 10);
  });

  test('surfaces a NetworkFailure on connection error', () async {
    when(() => dio.get<Map<String, dynamic>>(
          any(),
          queryParameters: any(named: 'queryParameters'),
        )).thenThrow(DioException(
      requestOptions: RequestOptions(path: ''),
      type: DioExceptionType.connectionError,
    ));

    final res = await repo.fetchAnimation(const LatLng(0, 0));
    expect(res.errorOrNull, isA<NetworkFailure>());
  });

  test('absoluteUrl leaves already-absolute URLs untouched', () {
    final src = PluvioBackendSource(dio: dio, baseUrl: 'https://pluvio.appmire.be');
    expect(src.absoluteUrl('https://cdn.example/x.png'), 'https://cdn.example/x.png');
    expect(src.absoluteUrl('/v1/overlay/nowcast/5.png'),
        'https://pluvio.appmire.be/v1/overlay/nowcast/5.png');
  });
}

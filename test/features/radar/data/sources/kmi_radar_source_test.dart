import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:pluvio/core/networking/api_failure.dart';
import 'package:pluvio/features/radar/data/sources/kmi_radar_source.dart';

import '../../../../_helpers/load_fixture.dart';

class _MockDio extends Mock implements Dio {}

void main() {
  late _MockDio dio;
  late KmiRadarSource source;

  setUp(() {
    registerFallbackValue(Options());
    dio = _MockDio();
    source = KmiRadarSource(
      dio: dio,
      wmsUrl: 'https://example.test/wms',
      layer: 'RADAR.BE_COMPOSITE',
    );
  });

  Response<String> okXml(String body) {
    return Response<String>(
      requestOptions: RequestOptions(path: ''),
      data: body,
      statusCode: 200,
    );
  }

  test('extracts the time dimension for the configured layer', () async {
    when(() => dio.get<String>(
          any(),
          queryParameters: any(named: 'queryParameters'),
          options: any(named: 'options'),
        )).thenAnswer(
      (_) async => okXml(loadFixtureString('kmi_radar_capabilities_sample.xml')),
    );

    final result = await source.fetchCapabilities();

    expect(result.isOk, isTrue);
    expect(result.valueOrNull!.timeSteps.length, 9);
  });

  test('returns ParseFailure when the layer is not present', () async {
    when(() => dio.get<String>(
          any(),
          queryParameters: any(named: 'queryParameters'),
          options: any(named: 'options'),
        )).thenAnswer((_) async => okXml('<WMS_Capabilities></WMS_Capabilities>'));

    final result = await source.fetchCapabilities();
    expect(result.errorOrNull, isA<ParseFailure>());
  });

  test('tileTemplateForFrame binds TIME and layer correctly', () {
    final url = source.tileTemplateForFrame(DateTime.utc(2026, 5, 22, 10));
    expect(url.queryParameters['time'], '2026-05-22T10:00:00.000Z');
    expect(url.queryParameters['layers'], 'RADAR.BE_COMPOSITE');
    expect(url.queryParameters['request'], 'GetMap');
  });
}

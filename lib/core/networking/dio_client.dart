import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:talker_dio_logger/talker_dio_logger.dart';

import '../logging/logger.dart';

/// Centralised Dio factory. Tests can override [dioProvider] with a fake.
final dioProvider = Provider<Dio>(_buildDio);

Dio _buildDio(Ref ref) {
  final dio = Dio(
    BaseOptions(
      connectTimeout: const Duration(seconds: 10),
      receiveTimeout: const Duration(seconds: 15),
      sendTimeout: const Duration(seconds: 10),
      headers: {
        HttpHeaders.acceptHeader: 'application/json',
        HttpHeaders.userAgentHeader: _userAgent,
      },
      responseType: ResponseType.json,
    ),
  );

  dio.interceptors.addAll([
    _RetryInterceptor(maxRetries: 2),
    TalkerDioLogger(
      talker: AppLogger.talker,
      settings: const TalkerDioLoggerSettings(
        printRequestHeaders: false,
        printResponseHeaders: false,
        printResponseData: false,
      ),
    ),
  ]);

  ref.onDispose(dio.close);
  return dio;
}

const String _userAgent = 'Pluvio/0.1 (+https://github.com/appmire/pluvio)';

/// Lightweight exponential-backoff retry for idempotent GETs only.
/// We retry transient network / 5xx errors twice; everything else surfaces.
class _RetryInterceptor extends Interceptor {
  _RetryInterceptor({required this.maxRetries});

  final int maxRetries;

  @override
  Future<void> onError(DioException err, ErrorInterceptorHandler handler) async {
    final method = err.requestOptions.method.toUpperCase();
    final attempt = (err.requestOptions.extra['attempt'] as int?) ?? 0;
    final retriable = method == 'GET' && _isTransient(err) && attempt < maxRetries;

    if (!retriable) {
      handler.next(err);
      return;
    }

    await Future<void>.delayed(Duration(milliseconds: 250 * (1 << attempt)));
    final next = err.requestOptions.copyWith(extra: {
      ...err.requestOptions.extra,
      'attempt': attempt + 1,
    });

    try {
      final response = await Dio().fetch<dynamic>(next);
      handler.resolve(response);
    } on DioException catch (e) {
      handler.next(e);
    }
  }

  bool _isTransient(DioException e) {
    if (e.type == DioExceptionType.connectionError ||
        e.type == DioExceptionType.connectionTimeout ||
        e.type == DioExceptionType.receiveTimeout) {
      return true;
    }
    final status = e.response?.statusCode ?? 0;
    return status >= 500 && status < 600;
  }
}

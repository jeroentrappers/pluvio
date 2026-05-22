import 'package:dio/dio.dart';

/// Domain-level error type. Data sources translate Dio exceptions into one of
/// these; the rest of the app deals only with [ApiFailure].
sealed class ApiFailure implements Exception {
  const ApiFailure(this.message, {this.cause, this.stackTrace});

  final String message;
  final Object? cause;
  final StackTrace? stackTrace;

  factory ApiFailure.fromDio(DioException e) {
    return switch (e.type) {
      DioExceptionType.connectionTimeout ||
      DioExceptionType.sendTimeout ||
      DioExceptionType.receiveTimeout =>
        TimeoutFailure(cause: e, stackTrace: e.stackTrace),
      DioExceptionType.connectionError =>
        NetworkFailure(cause: e, stackTrace: e.stackTrace),
      DioExceptionType.badResponse => ServerFailure(
        statusCode: e.response?.statusCode ?? -1,
        body: e.response?.data,
        cause: e,
        stackTrace: e.stackTrace,
      ),
      DioExceptionType.cancel =>
        const _CancelledFailure(),
      DioExceptionType.badCertificate ||
      DioExceptionType.unknown =>
        UnknownFailure(cause: e, stackTrace: e.stackTrace),
    };
  }

  @override
  String toString() => '$runtimeType($message)';
}

final class NetworkFailure extends ApiFailure {
  const NetworkFailure({super.cause, super.stackTrace})
    : super('No internet connection.');
}

final class TimeoutFailure extends ApiFailure {
  const TimeoutFailure({super.cause, super.stackTrace})
    : super('The request timed out.');
}

final class ServerFailure extends ApiFailure {
  const ServerFailure({
    required this.statusCode,
    this.body,
    super.cause,
    super.stackTrace,
  }) : super('Server returned HTTP $statusCode.');

  final int statusCode;
  final Object? body;
}

final class ParseFailure extends ApiFailure {
  const ParseFailure({super.cause, super.stackTrace})
    : super('Could not parse the server response.');
}

final class UnknownFailure extends ApiFailure {
  const UnknownFailure({super.cause, super.stackTrace})
    : super('Unexpected error.');
}

final class _CancelledFailure extends ApiFailure {
  const _CancelledFailure() : super('Request was cancelled.');
}

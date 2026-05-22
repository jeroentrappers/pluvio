# Contributing to Pluvio

Thanks for thinking about contributing. A few ground rules so we keep moving:

## Before you open a PR

1. `fvm flutter pub get`
2. `dart format lib test integration_test`
3. `fvm flutter analyze --fatal-infos`
4. `fvm flutter test --coverage`

CI runs the same four checks; matching them locally saves a round-trip.

## Architecture conventions

- **One feature per folder** under `lib/features/`. Each feature owns
  `domain/`, `data/`, `application/`, `presentation/`. Domain depends on
  nothing; presentation depends on application; application depends on data
  through interfaces in domain.
- **No secrets in code.** Everything tunable lives in `lib/core/config/env.dart`
  behind `String.fromEnvironment`, populated via `--dart-define` at build time.
- **Repositories return `Result<T, ApiFailure>`.** UI matches on the result;
  we don't throw across the data boundary for expected failures.
- **Tests mirror lib/.** `lib/features/x/y.dart` → `test/features/x/y_test.dart`.

## Commit style

Conventional commits-ish, lowercase: `feat(radar): ...`, `fix(location): ...`,
`chore(deps): ...`. Keep the subject under ~70 characters; details belong in
the body and PR description.

## Code of conduct

Be kind. Disagree about code, not people.

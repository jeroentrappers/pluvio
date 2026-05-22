// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for English (`en`).
class AppLocalizationsEn extends AppLocalizations {
  AppLocalizationsEn([String locale = 'en']) : super(locale);

  @override
  String get appTitle => 'Pluvio';

  @override
  String get refresh => 'Refresh';

  @override
  String get locationError => 'Couldn\'t determine your location.';

  @override
  String get radarError => 'Couldn\'t fetch radar data.';

  @override
  String get nowcastError => 'Couldn\'t fetch the nowcast.';

  @override
  String get nowcastDry => 'No rain expected in the next two hours.';

  @override
  String get nowcastRaining => 'It\'s raining now.';

  @override
  String nowcastRainInMinutes(int minutes) {
    final intl.NumberFormat minutesNumberFormat = intl.NumberFormat.decimalPattern(localeName);
    final String minutesString = minutesNumberFormat.format(minutes);

    return 'Rain expected in $minutesString min.';
  }
}

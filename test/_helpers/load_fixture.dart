import 'dart:convert';
import 'dart:io';

/// Reads a fixture file relative to the test working directory. Centralised
/// so tests don't all hardcode the same path pattern.
String loadFixtureString(String name) {
  return File('test/_fixtures/$name').readAsStringSync();
}

Map<String, dynamic> loadFixtureJson(String name) {
  return jsonDecode(loadFixtureString(name)) as Map<String, dynamic>;
}

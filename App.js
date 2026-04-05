import { useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import {
  StyleSheet,
  Text,
  View,
  TextInput,
  TouchableOpacity,
  TouchableWithoutFeedback,
  Keyboard,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';

export default function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleSearch = () => {
    if (!query.trim()) return;
    Keyboard.dismiss();
    setLoading(true);
    // TODO: wire up to backend
    setTimeout(() => {
      setResult(`Results for "${query}" will appear here.`);
      setLoading(false);
    }, 1000);
  };

  return (
    <TouchableWithoutFeedback onPress={Keyboard.dismiss} accessible={false}>
      <KeyboardAvoidingView
        style={styles.container}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        <StatusBar style="light" />

        <View style={styles.topSpacer} />

        <Text style={styles.title}>Noise</Text>
        <Text style={styles.subtitle}>Cut through the noise. Get answers.</Text>

        <View style={styles.searchContainer}>
          <Ionicons name="search" size={22} color="#888" style={styles.searchIcon} />
          <TextInput
            style={styles.searchInput}
            placeholder="Ask anything..."
            placeholderTextColor="#666"
            value={query}
            onChangeText={setQuery}
            onSubmitEditing={handleSearch}
            returnKeyType="search"
            autoCorrect={false}
          />
          {query.length > 0 && (
            <TouchableOpacity onPress={() => setQuery('')} style={styles.clearButton}>
              <Ionicons name="close-circle" size={20} color="#666" />
            </TouchableOpacity>
          )}
        </View>

        <TouchableOpacity
          style={[styles.searchButton, !query.trim() && styles.searchButtonDisabled]}
          onPress={handleSearch}
          disabled={!query.trim()}
          activeOpacity={0.7}
        >
          <Text style={styles.searchButtonText}>Search</Text>
        </TouchableOpacity>

        {loading && <ActivityIndicator size="large" color="#4F8EF7" style={styles.loader} />}

        {result && !loading && (
          <View style={styles.resultContainer}>
            <Text style={styles.resultText}>{result}</Text>
          </View>
        )}

        <View style={styles.bottomSpacer} />
      </KeyboardAvoidingView>
    </TouchableWithoutFeedback>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0D0D0D',
    alignItems: 'center',
    paddingHorizontal: 24,
  },
  topSpacer: {
    flex: 0.35,
  },
  bottomSpacer: {
    flex: 0.65,
  },
  title: {
    fontSize: 48,
    fontWeight: '800',
    color: '#FFFFFF',
    letterSpacing: 2,
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#888',
    marginBottom: 40,
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1A1A1A',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#2A2A2A',
    paddingHorizontal: 16,
    width: '100%',
    height: 56,
  },
  searchIcon: {
    marginRight: 12,
  },
  searchInput: {
    flex: 1,
    fontSize: 17,
    color: '#FFFFFF',
    height: '100%',
  },
  clearButton: {
    padding: 4,
  },
  searchButton: {
    marginTop: 16,
    backgroundColor: '#4F8EF7',
    borderRadius: 12,
    paddingVertical: 14,
    paddingHorizontal: 48,
  },
  searchButtonDisabled: {
    opacity: 0.4,
  },
  searchButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  loader: {
    marginTop: 32,
  },
  resultContainer: {
    marginTop: 32,
    backgroundColor: '#1A1A1A',
    borderRadius: 12,
    padding: 20,
    width: '100%',
  },
  resultText: {
    color: '#CCCCCC',
    fontSize: 15,
    lineHeight: 22,
  },
});

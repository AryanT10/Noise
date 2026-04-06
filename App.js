import { useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import {
  StyleSheet,
  Text,
  View,
  TextInput,
  TouchableOpacity,
  Keyboard,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Linking,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';

const API_BASE =
  Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://localhost:8000';

export default function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSearch = async () => {
    if (!query.trim()) return;
    Keyboard.dismiss();
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/ask/full`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query.trim() }),
      });

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        throw new Error(errBody.detail || `Server error (${response.status})`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Unable to reach the server.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <StatusBar style="light" />

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={[
          styles.scrollContent,
          !result && !loading && !error && styles.scrollContentCentered,
        ]}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.header}>
          <Text style={styles.title}>Noise</Text>
          <Text style={styles.subtitle}>Cut through the noise. Get answers.</Text>
        </View>

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

        <View style={styles.buttonRow}>
          <TouchableOpacity
            style={[styles.searchButton, !query.trim() && styles.searchButtonDisabled]}
            onPress={handleSearch}
            disabled={!query.trim() || loading}
            activeOpacity={0.7}
          >
            <Text style={styles.searchButtonText}>
              {loading ? 'Searching…' : 'Search'}
            </Text>
          </TouchableOpacity>

          {(result || error) && !loading && (
            <TouchableOpacity
              style={styles.clearResultButton}
              onPress={() => { setResult(null); setError(null); }}
              activeOpacity={0.7}
            >
              <Ionicons name="trash-outline" size={18} color="#FF6B6B" />
              <Text style={styles.clearResultText}>Clear</Text>
            </TouchableOpacity>
          )}
        </View>

        {loading && <ActivityIndicator size="large" color="#4F8EF7" style={styles.loader} />}

        {error && !loading && (
          <View style={styles.errorContainer}>
            <Ionicons name="alert-circle" size={18} color="#FF6B6B" />
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        {result && !loading && (
          <View style={styles.resultsSection}>
            {/* Answer */}
            <View style={styles.resultCard}>
              <Text style={styles.sectionLabel}>Answer</Text>
              <Text style={styles.answerText}>{result.answer}</Text>
            </View>

            {/* Sources */}
            {result.sources?.length > 0 && (
              <View style={styles.resultCard}>
                <Text style={styles.sectionLabel}>Sources</Text>
                {result.sources.map((src) => (
                  <TouchableOpacity
                    key={src.number}
                    style={styles.sourceRow}
                    onPress={() => Linking.openURL(src.url)}
                  >
                    <Text style={styles.sourceBadge}>[{src.number}]</Text>
                    <Text style={styles.sourceTitle} numberOfLines={1}>
                      {src.title}
                    </Text>
                    <Ionicons name="open-outline" size={14} color="#4F8EF7" />
                  </TouchableOpacity>
                ))}
              </View>
            )}

            {/* Consensus */}
            {result.consensus_groups?.length > 0 && (
              <View style={styles.resultCard}>
                <Text style={styles.sectionLabel}>Consensus</Text>
                {result.consensus_groups.map((g, i) => (
                  <View key={i} style={styles.consensusRow}>
                    <Text style={styles.consensusText}>
                      {g.canonical_claim}
                    </Text>
                    <Text style={styles.consensusMeta}>
                      {g.agreement_count} source{g.agreement_count !== 1 ? 's' : ''} agree
                    </Text>
                  </View>
                ))}
              </View>
            )}

            {/* Disagreements */}
            {result.disagreements?.length > 0 && (
              <View style={styles.resultCard}>
                <Text style={[styles.sectionLabel, { color: '#FF9F43' }]}>
                  Disagreements
                </Text>
                {result.disagreements.map((d, i) => (
                  <Text key={i} style={styles.bulletText}>• {d}</Text>
                ))}
              </View>
            )}

            {/* Uncertainties */}
            {result.uncertainties?.length > 0 && (
              <View style={styles.resultCard}>
                <Text style={[styles.sectionLabel, { color: '#A0A0A0' }]}>
                  Uncertainties
                </Text>
                {result.uncertainties.map((u, i) => (
                  <Text key={i} style={styles.bulletText}>• {u}</Text>
                ))}
              </View>
            )}
          </View>
        )}

        <View style={{ height: 60 }} />
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0D0D0D',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: 24,
    paddingTop: 60,
    alignItems: 'center',
  },
  scrollContentCentered: {
    flexGrow: 1,
    justifyContent: 'center',
    paddingTop: 0,
  },
  header: {
    alignItems: 'center',
    marginBottom: 40,
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
    marginBottom: 0,
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
  buttonRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    width: '100%',
    justifyContent: 'center',
  },
  clearResultButton: {
    marginTop: 16,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1A1A1A',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#2A2A2A',
    paddingVertical: 14,
    paddingHorizontal: 20,
    gap: 6,
  },
  clearResultText: {
    color: '#FF6B6B',
    fontSize: 15,
    fontWeight: '600',
  },
  loader: {
    marginTop: 32,
    marginBottom: 12,
  },
  resultsSection: {
    width: '100%',
    marginTop: 20,
  },
  errorContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 20,
    backgroundColor: '#1A1A1A',
    borderRadius: 10,
    padding: 14,
    width: '100%',
    gap: 8,
  },
  errorText: {
    color: '#FF6B6B',
    fontSize: 14,
    flex: 1,
  },
  resultCard: {
    backgroundColor: '#1A1A1A',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  sectionLabel: {
    color: '#4F8EF7',
    fontSize: 13,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 10,
  },
  answerText: {
    color: '#E0E0E0',
    fontSize: 15,
    lineHeight: 24,
  },
  sourceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    gap: 8,
  },
  sourceBadge: {
    color: '#4F8EF7',
    fontSize: 13,
    fontWeight: '700',
    width: 28,
  },
  sourceTitle: {
    color: '#CCCCCC',
    fontSize: 14,
    flex: 1,
  },
  consensusRow: {
    paddingVertical: 6,
  },
  consensusText: {
    color: '#D0D0D0',
    fontSize: 14,
    lineHeight: 20,
  },
  consensusMeta: {
    color: '#666',
    fontSize: 12,
    marginTop: 2,
  },
  bulletText: {
    color: '#BBBBBB',
    fontSize: 14,
    lineHeight: 22,
    paddingVertical: 2,
  },
});

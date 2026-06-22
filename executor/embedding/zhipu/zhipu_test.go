package zhipu

import (
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/omniX-dev/Omni-link/executor/embedding"
)

var loadOnce sync.Once

func tryLoadEnv() {
	loadOnce.Do(func() {
		paths := []string{".env", "../.env", "../../.env", "../../../.env"}
		for _, p := range paths {
			data, err := os.ReadFile(p)
			if err != nil {
				continue
			}
			for _, line := range strings.Split(string(data), "\n") {
				line = strings.TrimSpace(line)
				if line == "" || strings.HasPrefix(line, "#") {
					continue
				}
				parts := strings.SplitN(line, "=", 2)
				if len(parts) == 2 {
					os.Setenv(strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1]))
				}
			}
			break
		}
	})
}

func zhipuKey(t *testing.T) string {
	t.Helper()
	tryLoadEnv()
	k := os.Getenv("ZHIPU_API_KEY")
	if k == "" {
		t.Skip("ZHIPU_API_KEY not set")
	}
	return k
}

// channelMock provides GetName/GetBaseURL/GetAPIKey for testing.
type channelMock struct {
	name   string
	baseURL string
	apiKey string
}

func (m *channelMock) GetName() string  { return m.name }
func (m *channelMock) GetBaseURL() string { return m.baseURL }
func (m *channelMock) GetAPIKey() string  { return m.apiKey }

func TestZhipuEmbedding_Embed(t *testing.T) {
	key := zhipuKey(t)

	exec := &ZhipuEmbeddingExecutor{}
	exec.Init(&channelMock{
		name:   "zhipu-test",
		apiKey: key,
	})

	req := &embedding.EmbeddingRequest{
		Model:  "embedding-3",
		Input:  "Hello, world! Testing Zhipu embedding.",
	}

	resp, err := exec.Embed(req)
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}

	if resp == nil {
		t.Fatal("Embed returned nil response")
	}
	if resp.Object != "list" {
		t.Errorf("Object = %q, want \"list\"", resp.Object)
	}
	if len(resp.Data) == 0 {
		t.Fatal("Embed returned 0 data items")
	}
	if len(resp.Data[0].Embedding) == 0 {
		t.Fatal("Embedding vector is empty")
	}
	t.Logf("Model: %s", resp.Model)
	t.Logf("Vector dimensions: %d", len(resp.Data[0].Embedding))
	t.Logf("Usage: %d prompt tokens, %d total", resp.Usage.PromptTokens, resp.Usage.TotalTokens)
}

func TestZhipuEmbedding_EmbedBatch(t *testing.T) {
	key := zhipuKey(t)

	exec := &ZhipuEmbeddingExecutor{}
	exec.Init(&channelMock{
		name:   "zhipu-test",
		apiKey: key,
	})

	req := &embedding.EmbeddingRequest{
		Model:  "embedding-3",
		Input:  []string{"First text for embedding.", "Second text for embedding.", "Third text for embedding."},
		Dimensions: 256,
	}

	resp, err := exec.Embed(req)
	if err != nil {
		t.Fatalf("Embed batch failed: %v", err)
	}

	if len(resp.Data) != 3 {
		t.Errorf("Got %d embedding vectors, want 3", len(resp.Data))
	}
	for i, d := range resp.Data {
		if len(d.Embedding) > 256 {
			t.Errorf("Data[%d] has %d dims (requested 256)", i, len(d.Embedding))
		}
	}
	t.Logf("Model: %s", resp.Model)
	t.Logf("Batch vectors: %d x %d", len(resp.Data), len(resp.Data[0].Embedding))
	t.Logf("Usage: %d prompt tokens, %d total", resp.Usage.PromptTokens, resp.Usage.TotalTokens)
}

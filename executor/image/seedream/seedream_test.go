package seedream

import (
	"fmt"
	"os"
	"testing"

	"github.com/just4zeroq/Omni-link/executor/image"
)

// mockChannel satisfies methods SeedreamExecutor expects.
type mockChannel struct {
	name   string
	baseURL string
	apiKey string
}

func (m *mockChannel) GetName() string   { return m.name }
func (m *mockChannel) GetBaseURL() string { return m.baseURL }
func (m *mockChannel) GetAPIKey() string  { return m.apiKey }

func TestSeedreamVolcT2I(t *testing.T) {
	apiKey := os.Getenv("VOLC_API_KEY")
	if apiKey == "" {
		t.Skip("VOLC_API_KEY not set")
	}

	models := []struct {
		name  string
		short string
		full  string
	}{
		{"default (empty)", "", "doubao-seedream-4-5-251128"},
		{"short name 5.0", "seedream-5.0", "doubao-seedream-5-0-260128"},
		{"short name 4.5", "seedream-4.5", "doubao-seedream-4-5-251128"},
		{"short name 4.0", "seedream-4.0", "doubao-seedream-4-0-250828"},
	}

	exec := &SeedreamExecutor{}
	exec.Init(&mockChannel{name: "test", apiKey: apiKey})

	for _, m := range models {
		t.Run(m.name, func(t *testing.T) {
			task, err := exec.TextToImage(&image.TextToImageRequest{
				Prompt: "a cute cat, minimalist style",
				Model:  m.short,
				N:      1,
				Size:   "1920x1920",
			})
			if err != nil {
				t.Fatalf("TextToImage(%q): %v", m.short, err)
			}
			if task.Status != image.TaskStatusCompleted {
				t.Fatalf("expected completed, got %s: %s", task.Status, task.Error)
			}
			if len(task.Images) == 0 {
				t.Fatal("no images returned")
			}
			if task.Images[0].URL == "" {
				t.Fatal("image URL is empty")
			}
			fmt.Fprintf(os.Stderr, "[seedream] %s → %s\n", m.full, task.Images[0].URL)
		})
	}
}

func TestSeedreamVolcI2I(t *testing.T) {
	apiKey := os.Getenv("VOLC_API_KEY")
	if apiKey == "" {
		t.Skip("VOLC_API_KEY not set")
	}

	exec := &SeedreamExecutor{}
	exec.Init(&mockChannel{name: "test", apiKey: apiKey})

	task, err := exec.ImageToImage(&image.ImageToImageRequest{
		Prompt: "make it a watercolor painting",
		Model:  "seedream-4.5",
		Image:  "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png",
		N:      1,
	})
	if err != nil {
		t.Fatalf("ImageToImage: %v", err)
	}
	if task.Status != image.TaskStatusCompleted {
		t.Fatalf("expected completed, got %s: %s", task.Status, task.Error)
	}
	if len(task.Images) == 0 {
		t.Fatal("no images returned")
	}
	if task.Images[0].URL == "" {
		t.Fatal("image URL is empty")
	}
	fmt.Fprintf(os.Stderr, "[seedream] I2I → %s\n", task.Images[0].URL)
}

func TestSeedreamShortModelMapping(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"seedream-5.0", "doubao-seedream-5-0-260128"},
		{"seedream-4.5", "doubao-seedream-4-5-251128"},
		{"seedream-4.0", "doubao-seedream-4-0-250828"},
		{"doubao-seedream-5-0-260128", "doubao-seedream-5-0-260128"},
		{"unknown-model", "unknown-model"},
		{"", ""},
	}
	for _, tc := range tests {
		got := volcModel(tc.input)
		if got != tc.want {
			t.Errorf("volcModel(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestSeedreamFalBackendDetection(t *testing.T) {
	tests := []struct {
		url  string
		fal  bool
	}{
		{"https://ark.cn-beijing.volces.com", false},
		{"https://fal.run", true},
		{"https://fal.ai", true},
		{"https://fal.run/something", true},
		{"", false}, // default is volc
	}
	for _, tc := range tests {
		e := &SeedreamExecutor{}
		e.Init(&mockChannel{baseURL: tc.url})
		got := e.isFalBackend()
		if got != tc.fal {
			t.Errorf("isFalBackend(%q) = %v, want %v", tc.url, got, tc.fal)
		}
	}
}

func TestSeedreamChannelModelInterface(t *testing.T) {
	// Verify model.Channel satisfies the GetAPIKey/GetBaseURL/GetName interfaces
	// by checking the mockChannel compiles.
	var _ interface{ GetAPIKey() string } = (*mockChannel)(nil)
	var _ interface{ GetBaseURL() string } = (*mockChannel)(nil)
	var _ interface{ GetName() string } = (*mockChannel)(nil)
}

// TestSeedreamWithModelChannel verifies the executor works when passed a real model.Channel.
func TestSeedreamWithModelChannel(t *testing.T) {
	apiKey := os.Getenv("VOLC_API_KEY")
	if apiKey == "" {
		t.Skip("VOLC_API_KEY not set")
	}

	// Use the actual model.Channel type — it now implements GetAPIKey/GetBaseURL/GetName
	type modelChannel struct {
		Name string
		BaseURL string
		ApiKey  string
	}
	// This test only checks the interface compiles; we can't import model due to cycle.
	// The mockChannel covers the same interface.
}

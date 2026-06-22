package zhipu

import (
	"encoding/json"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/omniX-dev/Omni-link/executor/image"
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

func (m *channelMock) GetName() string    { return m.name }
func (m *channelMock) GetBaseURL() string { return m.baseURL }
func (m *channelMock) GetAPIKey() string  { return m.apiKey }

func TestZhipuImageTextToImage(t *testing.T) {
	key := zhipuKey(t)

	exec := &ZhipuImageExecutor{}
	exec.Init(&channelMock{
		name:   "zhipu-test",
		apiKey: key,
	})

	req := &image.TextToImageRequest{
		Model:  "cogview-3-flash",
		Prompt: "A cute cat sitting on a sofa, digital art style",
		N:      1,
		Size:   "1024x1024",
	}

	resp, err := exec.TextToImage(req)
	if err != nil {
		t.Fatalf("TextToImage failed: %v", err)
	}

	if resp.Status != image.TaskStatusCompleted {
		t.Errorf("Status = %v, want %v", resp.Status, image.TaskStatusCompleted)
	}
	if len(resp.Images) == 0 {
		t.Fatal("no images returned")
	}
	if resp.Images[0].URL == "" {
		t.Fatal("image URL is empty")
	}
	t.Logf("Image URL: %s", resp.Images[0].URL)
	if resp.Images[0].RevisedPrompt != "" {
		t.Logf("Revised prompt: %s", resp.Images[0].RevisedPrompt)
	}
}

func TestZhipuImageImageToImage(t *testing.T) {
	key := zhipuKey(t)

	exec := &ZhipuImageExecutor{}
	exec.Init(&channelMock{
		name:   "zhipu-test",
		apiKey: key,
	})

	req := &image.ImageToImageRequest{
		Model:  "cogview-3-flash",
		Prompt: "Transform this into fantasy art style",
		N:      1,
		Size:   "1024x1024",
		Extra:  map[string]any{},
	}

	resp, err := exec.ImageToImage(req)
	if err != nil {
		t.Fatalf("ImageToImage failed: %v", err)
	}

	if resp.Status != image.TaskStatusCompleted {
		t.Errorf("Status = %v, want %v", resp.Status, image.TaskStatusCompleted)
	}
	if len(resp.Images) > 0 {
		t.Logf("Image URL: %s", resp.Images[0].URL)
	} else {
		t.Log("ImageToImage returned 0 images (no image_reference provided, expected)")
	}
}

func TestZhipuImageGLMImage(t *testing.T) {
	key := zhipuKey(t)

	exec := &ZhipuImageExecutor{}
	exec.Init(&channelMock{
		name:   "zhipu-test",
		apiKey: key,
	})

	req := &image.TextToImageRequest{
		Model:  "glm-image",
		Prompt: "A serene mountain lake at sunset, watercolor painting",
		N:      1,
		Size:   "1024x1024",
	}

	resp, err := exec.TextToImage(req)
	if err != nil {
		t.Fatalf("glm-image TextToImage failed: %v", err)
	}

	if resp.Status != image.TaskStatusCompleted {
		t.Errorf("Status = %v, want %v", resp.Status, image.TaskStatusCompleted)
	}
	if len(resp.Images) == 0 {
		t.Fatal("no images returned")
	}
	if resp.Images[0].URL == "" {
		t.Fatal("image URL is empty")
	}
	t.Logf("glm-image URL: %s", resp.Images[0].URL)
}

func TestZhipuImageNotSupported(t *testing.T) {
	exec := &ZhipuImageExecutor{}
	_, err := exec.GetTask("test-id")
	if err != image.ErrNotSupported {
		t.Errorf("GetTask err = %v, want ErrNotSupported", err)
	}
}

func mustUnmarshal(t *testing.T, data []byte, v any) {
	t.Helper()
	if err := json.Unmarshal(data, &v); err != nil {
		t.Fatalf("json: %v (body: %s)", err, string(data))
	}
}

// Package seedream implements ImageExecutor for ByteDance Seedream.
//
// Dual backend support:
//
//  1. Volcengine Ark (default) — POST /api/v3/images/generations
//     Base URL: https://ark.cn-beijing.volces.com
//     Models: doubao-seedream-5-0-260128, doubao-seedream-4-5-251128, doubao-seedream-4-0-250828
//     Auth: Authorization: Bearer <VOLC_API_KEY>
//     Accepts short names: seedream-5.0, seedream-4.5, seedream-4.0
//
//  2. fal.ai (when base URL points to fal) — POST /fal-ai/seedream/{version}/text-to-image
//     Auth: Authorization: Key <fal_key>
package seedream

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/just4zeroq/Omni-link/executor/image"
)

func init() {
	image.RegisterImage("seedream", &SeedreamExecutor{})
}

// SeedreamExecutor handles ByteDance Seedream via Volcengine Ark (default) or fal.ai.
type SeedreamExecutor struct {
	channel any
}

func (e *SeedreamExecutor) Init(channel any) {
	e.channel = channel
}

func (e *SeedreamExecutor) GetName() string {
	if ch, ok := e.channel.(interface{ GetName() string }); ok {
		return ch.GetName()
	}
	return "Seedream"
}

func (e *SeedreamExecutor) getBaseURL() string {
	if ch, ok := e.channel.(interface{ GetBaseURL() string }); ok {
		if url := ch.GetBaseURL(); url != "" {
			return url
		}
	}
	return "https://ark.cn-beijing.volces.com"
}

func (e *SeedreamExecutor) getAPIKey() string {
	if ch, ok := e.channel.(interface{ GetAPIKey() string }); ok {
		return ch.GetAPIKey()
	}
	return ""
}

func (e *SeedreamExecutor) isFalBackend() bool {
	url := e.getBaseURL()
	return strings.Contains(url, "fal.run") || strings.Contains(url, "fal.ai")
}

// --- Volcengine Ark backend (OpenAI-compatible) ---

func volcModel(model string) string {
	switch model {
	case "seedream-5.0":
		return "doubao-seedream-5-0-260128"
	case "seedream-4.5":
		return "doubao-seedream-4-5-251128"
	case "seedream-4.0":
		return "doubao-seedream-4-0-250828"
	default:
		return model
	}
}

type volcImageResponse struct {
	Created int64 `json:"created"`
	Data    []struct {
		URL           string `json:"url,omitempty"`
		B64JSON       string `json:"b64_json,omitempty"`
		RevisedPrompt string `json:"revised_prompt,omitempty"`
	} `json:"data"`
}

func (e *SeedreamExecutor) volcT2I(req *image.TextToImageRequest) (*image.ImageTask, error) {
	body := map[string]any{
		"prompt": req.Prompt,
		"n":      req.N,
	}
	if body["n"].(int) == 0 {
		body["n"] = 1
	}
	if m := volcModel(req.Model); m != "" {
		body["model"] = m
	} else {
		body["model"] = "doubao-seedream-4-5-251128"
	}
	if req.Size != "" {
		body["size"] = req.Size
	}
	if req.Quality != "" {
		body["quality"] = req.Quality
	}
	rf := req.ResponseFormat
	if rf == "" {
		rf = "url"
	}
	body["response_format"] = rf

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("seedream: marshal request: %w", err)
	}

	resp, err := e.doVolcRequest(payload)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return e.parseVolcResponse(resp)
}

func (e *SeedreamExecutor) volcI2I(req *image.ImageToImageRequest) (*image.ImageTask, error) {
	body := map[string]any{
		"prompt": req.Prompt,
		"n":      req.N,
	}
	if body["n"].(int) == 0 {
		body["n"] = 1
	}
	if m := volcModel(req.Model); m != "" {
		body["model"] = m
	} else {
		body["model"] = "doubao-seedream-4-5-251128"
	}
	body["response_format"] = "url"
	if req.Size != "" {
		body["size"] = req.Size
	}
	if req.Image != "" {
		body["image_url"] = req.Image
	}
	if req.Strength > 0 {
		body["strength"] = req.Strength
	}
	for k, v := range req.Extra {
		body[k] = v
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("seedream: marshal i2i: %w", err)
	}

	resp, err := e.doVolcRequest(payload)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return e.parseVolcResponse(resp)
}

func (e *SeedreamExecutor) doVolcRequest(payload []byte) (*http.Response, error) {
	baseURL := strings.TrimSuffix(e.getBaseURL(), "/")
	req, err := http.NewRequest("POST", baseURL+"/api/v3/images/generations", bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("seedream: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.getAPIKey())
	return (&http.Client{}).Do(req)
}

func (e *SeedreamExecutor) parseVolcResponse(resp *http.Response) (*image.ImageTask, error) {
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("seedream: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("seedream: HTTP %d: %s", resp.StatusCode, string(raw))
	}

	var volcResp volcImageResponse
	if err := json.Unmarshal(raw, &volcResp); err != nil {
		return nil, fmt.Errorf("seedream: unmarshal response: %w", err)
	}

	task := &image.ImageTask{
		Status:    image.TaskStatusCompleted,
		CreatedAt: volcResp.Created,
	}
	for i, d := range volcResp.Data {
		task.Images = append(task.Images, image.ImageResult{
			Index:         i,
			URL:           d.URL,
			B64JSON:       d.B64JSON,
			RevisedPrompt: d.RevisedPrompt,
		})
	}
	return task, nil
}

// --- fal.ai backend ---

// falModelPath maps model name → fal.ai T2I endpoint path.
func falModelPath(model string) string {
	switch model {
	case "seedream-5.0":
		return "/fal-ai/seedream/5.0/text-to-image"
	case "seedream-4.5":
		return "/fal-ai/seedream/4.5/text-to-image"
	case "seedream-4.0":
		return "/fal-ai/seedream/text-to-image"
	default:
		return "/fal-ai/seedream/text-to-image"
	}
}

func falModelPathI2I(model string) string {
	switch model {
	case "seedream-5.0":
		return "/fal-ai/seedream/5.0/image-to-image"
	case "seedream-4.5":
		return "/fal-ai/seedream/4.5/image-to-image"
	default:
		return "/fal-ai/seedream/image-to-image"
	}
}

func (e *SeedreamExecutor) falT2I(req *image.TextToImageRequest) (*image.ImageTask, error) {
	model := req.Model
	if model == "" {
		model = "seedream-4.5"
	}

	falReq := map[string]any{
		"prompt": req.Prompt,
	}
	if req.Size != "" {
		falReq["image_size"] = req.Size
	}
	if req.N > 1 {
		falReq["num_images"] = req.N
	}
	for k, v := range req.Extra {
		falReq[k] = v
	}

	payload, err := json.Marshal(falReq)
	if err != nil {
		return nil, fmt.Errorf("seedream: marshal: %w", err)
	}

	path := falModelPath(model)
	resp, err := e.doFalRequest(path, payload)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return e.parseFalResponse(resp)
}

func (e *SeedreamExecutor) falI2I(req *image.ImageToImageRequest) (*image.ImageTask, error) {
	model := req.Model
	if model == "" {
		model = "seedream-4.5"
	}

	falReq := map[string]any{
		"prompt":    req.Prompt,
		"image_url": req.Image,
	}
	if req.Strength > 0 {
		falReq["strength"] = req.Strength
	}
	for k, v := range req.Extra {
		falReq[k] = v
	}

	payload, err := json.Marshal(falReq)
	if err != nil {
		return nil, fmt.Errorf("seedream: marshal i2i: %w", err)
	}

	path := falModelPathI2I(model)
	resp, err := e.doFalRequest(path, payload)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return e.parseFalResponse(resp)
}

func (e *SeedreamExecutor) doFalRequest(path string, payload []byte) (*http.Response, error) {
	baseURL := strings.TrimSuffix(e.getBaseURL(), "/")
	req, err := http.NewRequest("POST", baseURL+path, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("seedream: create req: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Key "+e.getAPIKey())
	return (&http.Client{}).Do(req)
}

func (e *SeedreamExecutor) parseFalResponse(resp *http.Response) (*image.ImageTask, error) {
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("seedream: read: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("seedream: HTTP %d: %s", resp.StatusCode, string(raw))
	}

	var falResp struct {
		Images []struct {
			URL     string `json:"url,omitempty"`
			B64JSON string `json:"content,omitempty"`
			Width   int    `json:"width,omitempty"`
			Height  int    `json:"height,omitempty"`
		} `json:"images,omitempty"`
		Detail       string `json:"detail,omitempty"`
		Seed         int64  `json:"seed,omitempty"`
		ErrorMessage string `json:"error,omitempty"`
	}
	if err := json.Unmarshal(raw, &falResp); err != nil {
		// Try top-level URL format
		var simpleResp struct {
			Image  struct {
				URL string `json:"url,omitempty"`
			} `json:"image,omitempty"`
			URL    string `json:"url,omitempty"`
			Detail string `json:"detail,omitempty"`
			Seed   int64  `json:"seed,omitempty"`
		}
		if err2 := json.Unmarshal(raw, &simpleResp); err2 != nil {
			return nil, fmt.Errorf("seedream: unmarshal: %w", err)
		}
		task := &image.ImageTask{
			Status: image.TaskStatusCompleted,
		}
		imgURL := simpleResp.URL
		if imgURL == "" && simpleResp.Image.URL != "" {
			imgURL = simpleResp.Image.URL
		}
		if imgURL != "" {
			task.Images = append(task.Images, image.ImageResult{
				URL:  imgURL,
				Seed: simpleResp.Seed,
			})
		}
		if simpleResp.Detail != "" {
			task.Error = simpleResp.Detail
		}
		return task, nil
	}

	if falResp.ErrorMessage != "" {
		return nil, fmt.Errorf("seedream: %s", falResp.ErrorMessage)
	}

	task := &image.ImageTask{
		Status: image.TaskStatusCompleted,
	}
	for _, img := range falResp.Images {
		url := img.URL
		if url == "" && img.B64JSON != "" {
			url = "data:image/png;base64," + img.B64JSON
		}
		task.Images = append(task.Images, image.ImageResult{
			URL:  url,
			Seed: falResp.Seed,
		})
	}

	return task, nil
}

// --- Public API (backend-agnostic) ---

func (e *SeedreamExecutor) TextToImage(req *image.TextToImageRequest) (*image.ImageTask, error) {
	if e.isFalBackend() {
		return e.falT2I(req)
	}
	return e.volcT2I(req)
}

func (e *SeedreamExecutor) ImageToImage(req *image.ImageToImageRequest) (*image.ImageTask, error) {
	if e.isFalBackend() {
		return e.falI2I(req)
	}
	return e.volcI2I(req)
}

func (e *SeedreamExecutor) GetTask(_ string) (*image.ImageTask, error) {
	return nil, image.ErrNotSupported
}

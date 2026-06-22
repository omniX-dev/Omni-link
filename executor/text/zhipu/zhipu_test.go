package zhipu

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/omniX-dev/Omni-link/executor/text"
	"github.com/omniX-dev/Omni-link/translator"
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

// ─── Text Executor Tests ───

const zhipuBaseURL = "https://open.bigmodel.cn/api/paas/v4"
const zhipuModel = "glm-4-flash"

func TestZhipuTextOpenAI(t *testing.T) {
	key := zhipuKey(t)
	info := &executor.RequestInfo{
		RequestID:      "zhipu-oa",
		UpstreamFormat: translator.FormatOpenAI,
		Model:          zhipuModel,
		ActualModelName: zhipuModel,
		InboundFormat:  translator.FormatOpenAI,
		ClientFormat:   translator.FormatOpenAI,
		ApiKey:         key,
		BaseURL:        zhipuBaseURL,
	}
	b, _ := json.Marshal(map[string]any{
		"model":    zhipuModel,
		"messages": []map[string]any{{"role": "user", "content": "Say hello in one word."}},
		"stream":   false,
	})
	resp, err := execReq(executor.GetByProvider("zhipu"), info, b)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	mustUnmarshal(t, resp, &m)
	choices, _ := m["choices"].([]any)
	if len(choices) == 0 {
		t.Fatalf("no choices: %s", string(resp))
	}
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	content, _ := msg["content"].(string)
	t.Logf("Response: %s", content)
}

func TestZhipuTextSystemMessage(t *testing.T) {
	key := zhipuKey(t)
	info := &executor.RequestInfo{
		RequestID:      "zhipu-sys",
		UpstreamFormat: translator.FormatOpenAI,
		Model:          zhipuModel,
		ActualModelName: zhipuModel,
		InboundFormat:  translator.FormatOpenAI,
		ClientFormat:   translator.FormatOpenAI,
		ApiKey:         key,
		BaseURL:        zhipuBaseURL,
	}
	b, _ := json.Marshal(map[string]any{
		"model": zhipuModel,
		"messages": []map[string]any{
			{"role": "system", "content": "Reply in ALL CAPS."},
			{"role": "user", "content": "Say hello"},
		},
		"stream": false,
	})
	resp, err := execReq(executor.GetByProvider("zhipu"), info, b)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	mustUnmarshal(t, resp, &m)
	choices, _ := m["choices"].([]any)
	if len(choices) == 0 {
		t.Fatalf("no choices: %s", string(resp))
	}
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	content, _ := msg["content"].(string)
	t.Logf("System msg response (expect caps): %s", content)
}

func TestZhipuTextStreamOpenAI(t *testing.T) {
	key := zhipuKey(t)
	info := &executor.RequestInfo{
		RequestID:      "zhipu-str",
		UpstreamFormat: translator.FormatOpenAI,
		Model:          zhipuModel,
		ActualModelName: zhipuModel,
		InboundFormat:  translator.FormatOpenAI,
		ClientFormat:   translator.FormatOpenAI,
		ApiKey:         key,
		BaseURL:        zhipuBaseURL,
		IsStream:       true,
	}
	b, _ := json.Marshal(map[string]any{
		"model":    zhipuModel,
		"messages": []map[string]any{{"role": "user", "content": "Count 1 to 5."}},
		"stream":   true,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var chunks [][]byte
	err := executor.ExecuteStream(ctx, executor.GetByProvider("zhipu"), info, b, func(chunk []byte) error {
		c := make([]byte, len(chunk))
		copy(c, chunk)
		chunks = append(chunks, c)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) == 0 {
		t.Fatal("no chunks received")
	}

	gotData := false
	gotDone := false
	full := string(bytes.Join(chunks, nil))
	for _, line := range strings.Split(full, "\n") {
		line = strings.TrimSpace(line)
		if line == "data: [DONE]" {
			gotDone = true
		} else if strings.HasPrefix(line, "data: ") {
			gotData = true
		}
	}
	if !gotData {
		t.Fatal("no data chunks received")
	}
	if !gotDone {
		t.Fatal("no [DONE] terminator")
	}
	t.Logf("Stream: %d chunks, %.2f KB", len(chunks), float64(len(bytes.Join(chunks, nil)))/1024)
}

func TestZhipuTextWithParams(t *testing.T) {
	key := zhipuKey(t)
	info := &executor.RequestInfo{
		RequestID:      "zhipu-params",
		UpstreamFormat: translator.FormatOpenAI,
		Model:          zhipuModel,
		ActualModelName: zhipuModel,
		InboundFormat:  translator.FormatOpenAI,
		ClientFormat:   translator.FormatOpenAI,
		ApiKey:         key,
		BaseURL:        zhipuBaseURL,
	}
	stop := []string{"."}
	b, _ := json.Marshal(map[string]any{
		"model":       zhipuModel,
		"messages":    []map[string]any{{"role": "user", "content": "Count 1 2 3"}},
		"temperature": 0.5,
		"top_p":       0.9,
		"stop":        stop,
	})
	resp, err := execReq(executor.GetByProvider("zhipu"), info, b)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	mustUnmarshal(t, resp, &m)
	choices, _ := m["choices"].([]any)
	if len(choices) == 0 {
		t.Fatal("no choices")
	}
	t.Logf("Params response: %s", string(resp))
}

func TestZhipuTextErrorBadKey(t *testing.T) {
	_ = zhipuKey(t) // skip if no key, but use bad key for actual test
	info := &executor.RequestInfo{
		RequestID:      "zhipu-err",
		UpstreamFormat: translator.FormatOpenAI,
		Model:          zhipuModel,
		ActualModelName: zhipuModel,
		InboundFormat:  translator.FormatOpenAI,
		ClientFormat:   translator.FormatOpenAI,
		ApiKey:         "sk-bad-key-that-will-fail",
		BaseURL:        zhipuBaseURL,
	}
	b, _ := json.Marshal(map[string]any{
		"model":    zhipuModel,
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})

	e := executor.GetByProvider("zhipu")
	status, _, err := execReqRaw(e, info, b)
	if err != nil {
		t.Fatalf("execReqRaw err: %v", err)
	}
	if status == 200 {
		t.Fatal("expected non-200 status for bad key")
	}
	t.Logf("Bad key status: %d (expected 401/403)", status)
}

// ─── helpers (mirror deepseek_test.go) ───

func execReq(e executor.Executor, info *executor.RequestInfo, body []byte) ([]byte, error) {
	status, data, err := execReqRaw(e, info, body)
	if err != nil {
		return nil, err
	}
	if status != 200 {
		return nil, fmt.Errorf("status %d: %s", status, string(data))
	}
	return data, nil
}

func execReqRaw(e executor.Executor, info *executor.RequestInfo, body []byte) (int, []byte, error) {
	up := info.UpstreamFormat
	if up == "" {
		up = translator.FormatOpenAI
	}
	conv, err := e.ConvertRequest(body, info.InboundFormat, up)
	if err != nil {
		return 0, nil, err
	}
	conv = e.RequestCustomize(conv, info)
	resp, err := e.DoRequest(info, bytes.NewReader(conv))
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	respBody = e.ResponseCustomize(respBody, info)
	convResp, err := e.ConvertResponse(respBody, up, info.ClientFormat)
	if err != nil {
		return resp.StatusCode, respBody, err
	}
	return resp.StatusCode, convResp, nil
}

func mustUnmarshal(t *testing.T, data []byte, v any) {
	t.Helper()
	if err := json.Unmarshal(data, &v); err != nil {
		t.Fatalf("json: %v (body: %s)", err, string(data))
	}
}

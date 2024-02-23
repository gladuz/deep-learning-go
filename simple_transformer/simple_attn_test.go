package transformer

import (
	"math"
	"testing"
)

func TestLayerNorm(t *testing.T) {
	x := Matrix{1, 4, []float32{1, 2, 3, 4}}
	weight := []float32{1, 2, 3, 4}
	want := []float32{-1.3416, -0.8944, 1.3416, 5.3665}

	x.LayerNorm(weight)
	got := x.Data

	eps := 1e-4
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > eps {
			t.Fatalf("wanted %f, got %f", want[i], got[i])
		}
	}
}

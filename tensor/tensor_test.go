package tensor

import (
	"testing"
)

func TestNewMatrix(t *testing.T) {
	a := NewTensor(nil, nil)
	if len(a.Data) != 0 {
		t.Error("Tensor is not initialized")
	}
}

func TestMatrixAt(t *testing.T) {
	t.Run("two dimensional", func(t *testing.T) {
		a := NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
		got := a.At(1, 0)
		want := 3.0
		if got != want {
			t.Errorf("wanted %f, got %f", want, got)
		}
	})
	t.Run("many dimensional", func(t *testing.T) {
		a := NewTensor([]float64{1, 2, 3, 4, 5, 6, 7, 8}, []int{2, 2, 2})
		got := a.At(1, 0, 1)
		want := 6.0
		if got != want {
			t.Errorf("wanted %f, got %f", want, got)
		}
	})
}

func TestMatrixView(t *testing.T) {
	t.Run("test view basic", func(t *testing.T) {
		a := NewTensor([]float64{1, 2, 3, 4, 5, 6, 7, 8}, []int{2, 2, 2})
		got := a.View(4, 2).At(2, 1)
		want := 6.0
		if got != want {
			t.Errorf("wanted %f, got %f", want, got)
		}
	})
	t.Run("test view storage", func(t *testing.T) {
		a := NewTensor([]float64{1, 2, 3, 4, 5, 6, 7, 8}, []int{2, 2, 2})
		b := a.View(4, 2)
		a.Set(12, 1, 0, 1)
		got := b.At(2, 1)
		want := 12.0
		if got != want {
			t.Errorf("wanted %f, got %f", want, got)
		}
	})
}

func TestDimSlice(t *testing.T) {
	a := NewTensor([]float64{1, 2, 3, 4, 5, 6, 7, 8}, []int{2, 2, 2})
	t.Run("test getting 0 dim", func(t *testing.T) {
		b := a.DimSlice(0, 1)
		/*
			tensor([[5, 6],
					[7, 8]])
		*/
		got := b.At(1, 1)
		want := 8.0
		if got != want {
			t.Errorf("wanted %f, got %f", want, got)

		}
	})
	t.Run("test getting 1 dim", func(t *testing.T) {
		b := a.DimSlice(1, 1)
		/*
			tensor([[3, 4],
					[7, 8]])
		*/
		got := b.At(1, 0)
		want := 7.0
		if got != want {
			t.Errorf("wanted %f, got %f", want, got)

		}
	})
	t.Run("test getting 2 dim", func(t *testing.T) {
		b := a.DimSlice(2, 1)
		/*
			tensor([[2, 4],
					[6, 8]])
		*/
		got := b.At(1, 0)
		want := 6.0
		if got != want {
			t.Errorf("wanted %f, got %f", want, got)

		}
	})
	t.Run("test getting 2 dim ind 0", func(t *testing.T) {
		b := a.DimSlice(2, 0)
		got := b.At(1, 0)
		/*
			tensor([[1, 3],
			    	[5, 7]])
		*/
		want := 5.0
		if got != want {
			t.Errorf("wanted %f, got %f", want, got)

		}
	})
}

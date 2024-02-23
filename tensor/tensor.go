package tensor

import (
	"log"
)

type Tensor struct {
	Sizes   []int
	Strides []int
	Offset  int
	Data    []float64
}

func NewTensor(data []float64, sizes []int) *Tensor {
	t := Tensor{}
	if len(data) == 0 {
		t.Data = make([]float64, 0)
		t.Sizes = make([]int, 0)
		t.Offset = 0
		t.Strides = make([]int, 0)
		return &t
	}
	t.Data = data
	t.Sizes = sizes
	t.Offset = 0

	t.Strides = calcualteStrides(sizes)
	return &t
}

func (t *Tensor) At(idx ...int) float64 {
	if len(idx) != len(t.Sizes) {
		log.Fatal("Too many indices to access")
	}
	ind := 0
	for i := 0; i < len(t.Sizes); i++ {
		ind += idx[i] * t.Strides[i]
	}
	if ind >= len(t.Data) {
		log.Fatal("The index is too high")
	}
	return t.Data[t.Offset+ind]
}

func (t *Tensor) Set(val float64, idx ...int) {
	ind := 0
	for i := 0; i < len(t.Sizes); i++ {
		ind += idx[i] * t.Strides[i]
	}
	t.Data[ind] = val
}

func (t *Tensor) View(idx ...int) *Tensor {
	if prod(idx) != len(t.Data) {
		log.Fatal("The dimensions do not match number of elements")
	}
	strides := calcualteStrides(idx)
	return &Tensor{
		Data:    t.Data,
		Sizes:   idx,
		Strides: strides,
		Offset:  0,
	}
}

// Returns the view along the dimension with ind
func (t *Tensor) DimSlice(dim, ind int) *Tensor {
	newSizes := make([]int, len(t.Sizes)-1)
	for i := range len(t.Sizes) - 1 {
		newSizes[i] = t.Sizes[i]
	}
	newStrides := make([]int, len(t.Strides)-1)
	ii := 0
	for i := 0; i < len(t.Strides); i++ {
		if i != dim {
			newStrides[ii] = t.Strides[i]
			ii++
		}
	}
	newOffset := t.Strides[dim] * ind
	return &Tensor{
		Data:    t.Data,
		Sizes:   newSizes,
		Strides: newStrides,
		Offset:  newOffset,
	}
}

func prod(idx []int) int {
	res := 1
	for i := range len(idx) {
		res *= idx[i]
	}
	return res
}

func calcualteStrides(sizes []int) []int {
	strides := make([]int, len(sizes))
	inter_stride := 1
	for i := len(strides) - 1; i >= 0; i-- {
		strides[i] = inter_stride
		inter_stride *= sizes[i]
	}
	return strides
}
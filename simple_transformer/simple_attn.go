package transformer

import (
	"encoding/binary"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
)

const (
	N_INP             = 4
	N_EMB             = 8
	N_CLASSES         = 3
	SEQ_LEN           = 8
	ATTN_DROPOUT      = 0.2
	ATTN_PROJ_DROPOUT = 0.2
)

var DATA_MEAN = [4]float32{5.8433, 3.0573, 3.7580, 1.1993}
var DATA_STD = [4]float32{0.8281, 0.4359, 1.7653, 0.7622}

type Matrix struct {
	Rows int
	Cols int
	Data []float32
}

type NormWeights struct {
	Dim   int
	Scale []float32
}

func MatMul(x Matrix, y Matrix) Matrix {
	if x.Cols != y.Rows {
		panic("Invalid matrix dimensions")
	}
	res := make([]float32, x.Rows*y.Cols)
	for i := 0; i < x.Rows; i++ {
		for j := 0; j < y.Cols; j++ {
			for k := 0; k < x.Cols; k++ {
				res[i*y.Cols+j] += x.Data[i*x.Cols+k] * y.Data[k*y.Cols+j]
			}
		}
	}
	return Matrix{
		Rows: x.Rows,
		Cols: y.Cols,
		Data: res,
	}
}

func (m *Matrix) T() Matrix {
	res := make([]float32, m.Rows*m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			res[j*m.Rows+i] = m.Data[i*m.Cols+j]
		}
	}
	m.Data = res
	m.Rows, m.Cols = m.Cols, m.Rows
	return *m
}

func (m *Matrix) Apply(f func(float32) float32) {
	for i := range m.Data {
		m.Data[i] = f(m.Data[i])
	}
}

type ModelWeights struct {
	InputW  Matrix
	AttnW   AttentionWeights
	OutputW Matrix
}

type AttentionWeights struct {
	QKV     Matrix
	OutProj Matrix
	Linear1 Matrix
	Linear2 Matrix
	Norm1   NormWeights
	Norm2   NormWeights
}

func RandomMatrix(n, d int) Matrix {
	rand_data := make([]float32, n*d)
	for i := range rand_data {
		rand_data[i] = float32(rand.NormFloat64() / math.Sqrt(float64(d)))
	}
	return Matrix{
		Rows: n,
		Cols: d,
		Data: rand_data,
	}
}

func NewModelWeights() ModelWeights {
	return ModelWeights{
		InputW: RandomMatrix(N_INP, N_EMB),
		AttnW: AttentionWeights{
			QKV:     RandomMatrix(N_EMB, N_EMB*3),
			OutProj: RandomMatrix(N_EMB, N_EMB),
			Linear1: RandomMatrix(N_EMB, N_EMB),
			Linear2: RandomMatrix(N_EMB, N_EMB),
			Norm1:   NormWeights{Scale: make([]float32, N_EMB)},
			Norm2:   NormWeights{Scale: make([]float32, N_EMB)},
		},
		OutputW: RandomMatrix(N_EMB, N_CLASSES),
	}
}

func Softmax(x Matrix) {
	n, d := x.Rows, x.Cols
	for i := 0; i < n; i++ {
		rowMax := x.Data[i*d]
		for j := 0; j < d; j++ {
			if x.Data[i*d+j] > rowMax {
				rowMax = x.Data[i*d+j]
			}
		}
		rowSum := float32(0)
		for j := 0; j < d; j++ {
			x.Data[i*d+j] = float32(math.Exp(float64(x.Data[i*d+j] - rowMax)))
			rowSum += x.Data[i*d+j]
		}
		for j := 0; j < d; j++ {
			x.Data[i*d+j] /= rowSum
		}
	}
}

func (m *Matrix) Add(b Matrix) {
	for i := range m.Data {
		m.Data[i] += b.Data[i]
	}
}

func (m *Matrix) Row(i int) []float32 {
	return m.Data[i*m.Cols : (i+1)*m.Cols]
}

func (m *Matrix) Argmax() int {
	max := m.Data[0]
	argmax := 0
	for i := 1; i < len(m.Data); i++ {
		if m.Data[i] > max {
			max = m.Data[i]
			argmax = i
		}
	}
	return argmax
}

func (x *Matrix) LayerNorm(weight []float32) {
	N, D := x.Rows, x.Cols
	for i := range N {
		row := x.Row(i)
		//Use unbiased std
		mean := float32(0)
		for _, v := range row {
			mean += v
		}
		mean /= float32(D)
		variance := float32(0)
		for _, v := range row {
			variance += (v - mean) * (v - mean)
		}
		variance = float32(math.Sqrt(float64(variance/float32(D) + 1e-5)))

		for j := range row {
			row[j] = (row[j] - mean) / variance * weight[j]
		}
	}
}

func SelfAttention(x Matrix, attnW AttentionWeights) Matrix {
	N, D := x.Rows, x.Cols
	qkv := MatMul(x, attnW.QKV)
	// Split qkv into q, k, v
	q := Matrix{Rows: N, Cols: D, Data: qkv.Data[:N*D]}
	k := Matrix{Rows: N, Cols: D, Data: qkv.Data[N*D : 2*N*D]}
	v := Matrix{Rows: N, Cols: D, Data: qkv.Data[2*N*D : 3*N*D]}

	attn := MatMul(q, k.T())
	attn.Apply(func(x float32) float32 {
		return x / float32(math.Sqrt(float64(D)))
	})
	Softmax(attn)
	attn = MatMul(attn, v)
	attn = MatMul(x, attnW.OutProj)

	attn.Add(x)
	attn.LayerNorm(attnW.Norm1.Scale)
	x = MatMul(attn, attnW.Linear1)
	x.Apply(func(x float32) float32 {
		return float32(math.Max(float64(x), 0))
	})
	x = MatMul(x, attnW.Linear2)
	x.Add(attn)
	x.LayerNorm(attnW.Norm2.Scale)

	return x
}

func loadDataset(filename string) ([][]float32, []string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, errors.New("can't open the file")
	}
	defer file.Close()

	reader := csv.NewReader(file)

	// Skip header
	reader.Read()

	var resultX [][]float32
	var resultY []string

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, errors.New("can't parse the csv")
		}
		recX := make([]float32, 4)
		for i := 1; i < 5; i++ {
			parsed, err := strconv.ParseFloat(record[i], 32)
			recX[i-1] = float32(parsed)
			if err != nil {
				return nil, nil, errors.New("can't parse the float from dataset")
			}
		}
		resultX = append(resultX, recX)
		resultY = append(resultY, record[5])
	}
	return resultX, resultY, err

}

func LoadPretrainedWeights(filename string) ModelWeights {
	// Format of the binary is as follows
	// (N_SHAPES, *N_SHAPES, DATA) * 8
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	weights := NewModelWeights()

	weights.InputW.Apply(func(float32) float32 {
		return 0
	})
	weight_data_list := []*[]float32{
		&weights.InputW.Data,
		&weights.AttnW.QKV.Data,
		&weights.AttnW.OutProj.Data,
		&weights.AttnW.Linear1.Data,
		&weights.AttnW.Linear2.Data,
		&weights.AttnW.Norm1.Scale,
		&weights.AttnW.Norm2.Scale,
		&weights.OutputW.Data,
	}
	for _, weight_data := range weight_data_list {
		err := binary.Read(file, binary.LittleEndian, weight_data)
		if err != nil {
			log.Fatal(err)
		}
	}

	if err != nil {
		log.Fatal(err)
	}
	//check if the end of the file is reached
	_, err = file.Read([]byte{1})
	if err != io.EOF {
		log.Fatal("File is not fully read")
	}

	return weights
}

func Normalize(x []float32) []float32 {
	for j := range x {
		x[j] = (x[j] - DATA_MEAN[j]) / DATA_STD[j]
	}
	return x
}

func Forward(x Matrix, weights ModelWeights) Matrix {
	Normalize(x.Data)
	x = MatMul(x, weights.InputW)
	x = SelfAttention(x, weights.AttnW)
	x = MatMul(x, weights.OutputW)
	Softmax(x)
	return x
}

func main() {
	weights := LoadPretrainedWeights("weights.bin")
	x, y, err := loadDataset("Iris.csv")
	if err != nil {
		log.Fatal(err)
	}
	target_names := map[string]int{
		"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2,
	}
	correct := 0
	for i := range x {
		inp := Matrix{Rows: 1, Cols: 4, Data: x[i]}
		out := Forward(inp, weights)
		pred := out.Argmax()
		if pred == target_names[y[i]] {
			correct += 1
		}
	}
	fmt.Println("Accuracy: ", float32(correct)/float32(len(x)))

}

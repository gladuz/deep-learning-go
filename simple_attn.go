package main

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type AttentionWeights struct{
	Q *mat.Dense
	K *mat.Dense
	V *mat.Dense
}

func ExpInplace(x *mat.VecDense){
	for i:=0; i<x.Len(); i++{
		x.SetVec(i, math.Exp(x.AtVec(i)))
	}
}

func Sum(x mat.Vector) float64{
	sum := 0.0
	for i:=0; i<x.Len(); i++{
		sum += x.AtVec(i)
	}
	return sum
}

func SoftmaxVec(x *mat.VecDense) *mat.VecDense{
	//Numerical stabilization
	max_x := mat.Max(x)
	for i:= range x.Len(){
		x.SetVec(i, x.AtVec(i) - max_x)
	}
	ExpInplace(x)
	sum_exps := Sum(x)
	result := mat.NewVecDense(x.Len(), nil)
	result.ScaleVec(1/sum_exps, x)
	return result
}

func SoftmaxMat(x *mat.Dense){
	for c := range x.RawMatrix().Cols{
		x.SetCol(c, SoftmaxVec(mat.VecDenseCopyOf(x.ColView(c))).RawVector().Data)
	}
}

func Relu(x *mat.VecDense) *mat.VecDense{
	result := mat.NewVecDense(x.Len(), nil)
	for i:=0; i<x.Len(); i++{
		result.SetVec(i, max(x.AtVec(i), 0.0))
	}
	return result
}

func (w *AttentionWeights)Init(n int, d int){
	w.Q = mat.NewDense(n, d, RandomMatrix(n, d))
	w.K = mat.NewDense(n, d, RandomMatrix(n, d))
	w.V = mat.NewDense(n, d, RandomMatrix(n, d))
}

func RandomMatrix(n int, d int) []float64 {
	rand_data := make([]float64, n*d)
	for i := range rand_data {
		rand_data[i] = rand.NormFloat64()
	}
	return rand_data
}

func Attention(x *mat.VecDense, attnW AttentionWeights) *mat.VecDense{
	q := mat.NewVecDense(attnW.Q.RawMatrix().Rows, nil)
	k := mat.NewVecDense(attnW.Q.RawMatrix().Rows, nil)
	v := mat.NewVecDense(attnW.Q.RawMatrix().Rows, nil)

	q.MulVec(attnW.Q, x)
	k.MulVec(attnW.K, x)
	v.MulVec(attnW.V, x)

	k_t := k.T()
	q_tt := q.T().T()
	
	attn := mat.NewDense(q.Len(), k.Len(), nil)
	scaler := math.Sqrt(float64(attnW.Q.RawMatrix().Rows))
	attn.Scale(scaler, attn)
	attn.Mul(q_tt, k_t)
	SoftmaxMat(attn)

	v.MulVec(attn, v)
	return v
}


func main() {
	d := mat.NewVecDense(3, []float64{-1,0.5,-0.6})
	//d = SoftmaxVec(d)
	attWeights := AttentionWeights{}
	attWeights.Init(5, 3)
	att_x := Attention(d, attWeights)
	att_x = Relu(att_x)
	

	//fmt.Printf("%v, %f\n", d, Sum(d))
	fmt.Println(att_x)
}

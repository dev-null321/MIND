#lang racket

(require "tensor.rkt")
(provide dense-forward mean-squared-error dense-backward relu relu-derivative initialize-fnn)

(define (relu x)
  (tensor (tensor-shape x) (for/vector ([v (tensor-data x)]) (max 0 v))))

(define (relu-derivative x)
  (tensor (tensor-shape x) (for/vector ([v (tensor-data x)]) (if (> v 0) 1 0))))

(define (dense-forward input weights biases)
  (let* ([mul-result (tensor-multiply input weights)]
         [mul-result-shape (tensor-shape mul-result)]
         [output-dim (cadr mul-result-shape)]
         [reshaped-biases (reshape-tensor biases (list output-dim))]
         [z (tensor-add mul-result reshaped-biases)]
         [activation-output (relu z)])
    (displayln (string-append "mul-result shape: " (format "~a" mul-result-shape)))
    (displayln (string-append "reshaped-biases shape: " (format "~a" (tensor-shape reshaped-biases))))
    activation-output))

(define (mean-squared-error y-true y-pred)
  (let* ([diff (tensor-subtract y-true y-pred)]
         [squared-diff (tensor-multiply diff diff)]
         [sum (apply + (vector->list (tensor-data squared-diff)))])
    (/ sum (length (vector->list (tensor-data y-true))))))

(define (dense-backward input weights biases output grad-output learning-rate)
  (let* ([grad-activation (relu-derivative output)]
         [grad-z (tensor-multiply grad-output grad-activation)]
         [grad-weights (tensor-multiply (transpose input) grad-z)]
         [grad-biases (tensor (list (vector-length (tensor-data biases)))
                              (for/vector ([j (vector-length (tensor-data biases))])
                                (apply + (for/list ([i (car (tensor-shape grad-z))])
                                           (vector-ref (tensor-data grad-z) (+ (* i (vector-length (tensor-data biases))) j))))))]
         [grad-input (tensor-multiply grad-z (transpose weights))])
    (let* ([new-weights (tensor-subtract weights (scalar-multiply grad-weights learning-rate))]
           [new-biases (tensor-subtract biases (scalar-multiply grad-biases learning-rate))])
      (values new-weights new-biases grad-input))))

(define (initialize-fnn batch-size input-dim output-dim)
  (let* ([input-data (make-list (* batch-size input-dim) 0)]
         [input-tensor (create-tensor (list batch-size input-dim) input-data)]
         [weight-shape (list input-dim output-dim)]
         [bias-shape (list output-dim)]  ; Ensure same size as output dim
         [weights (random-tensor weight-shape 1.0)]
         [biases (random-tensor bias-shape 1.0)])
    (values input-tensor weights biases)))
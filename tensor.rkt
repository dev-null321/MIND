#lang racket

(provide tensor create-tensor tensor-add tensor-subtract tensor-multiply print-tensor random-tensor reshape-tensor 
  transpose scalar-multiply dot-product tensor-shape tensor-data tensor-ref)

(struct tensor (shape data) #:transparent)

(define (create-tensor shape data)
  (let ((vec-data (if (vector? data) data (list->vector data))))
    (cond
      [(= (apply * shape) (vector-length vec-data))
       (tensor shape vec-data)]
      [else
       (begin
         (println "Error: Data does not match, please check the size")
         #f)])))

(define (print-tensor t)
  (let ([shape (tensor-shape t)]
        [data (tensor-data t)])
    (cond
      [(= (length shape) 1)
       (display "[")
       (for ([i (in-range (car shape))])
         (display (vector-ref data i))
         (display " "))
       (display "]")
       (newline)]
      [(= (length shape) 2)
       (for ([i (in-range (car shape))])
         (display "[")
         (for ([j (in-range (cadr shape))])
           (display (vector-ref data (+ (* i (cadr shape)) j)))
           (display " "))
         (display "]")
         (newline))]
      [else
       (error "print-tensor: unsupported tensor shape")])))

(define (random-tensor shape range)
  (let* ((size (apply * shape))
         (max-value (inexact->exact (floor (* range 10000)))))
    (tensor shape
            (for/vector ([i size])
              (/ (random max-value) 10000.0)))))

(define (reshape-tensor t new-shape)
  (let ([original-size (apply * (tensor-shape t))]
        [new-size (apply * new-shape)])
    (if (= original-size new-size)
        (tensor new-shape (tensor-data t))
        (error "New shape must have the same number of elements as the original shape"))))

(define (tensor-add t1 t2)
  (let ([shape1 (tensor-shape t1)]
        [shape2 (tensor-shape t2)])
    (cond
      [(equal? shape1 shape2)
       (tensor shape1 (for/vector ([i (vector-length (tensor-data t1))])
                        (+ (vector-ref (tensor-data t1) i)
                           (vector-ref (tensor-data t2) i))))]
      [(= (length shape1) 1)
       (let ([scalar-val (vector-ref (tensor-data t1) 0)])
         (tensor shape2 (for/vector ([i (vector-length (tensor-data t2))])
                          (+ scalar-val (vector-ref (tensor-data t2) i)))))]
      [(= (length shape2) 1)
       (let ([scalar-val (vector-ref (tensor-data t2) 0)])
         (tensor shape1 (for/vector ([i (vector-length (tensor-data t1))])
                          (+ (vector-ref (tensor-data t1) i) scalar-val))))]
      [else
       (error "Tensors must have the same shape for addition")])))

(define (tensor-subtract t1 t2)
  (let ([shape1 (tensor-shape t1)]
        [shape2 (tensor-shape t2)])
    (displayln (string-append "Subtracting tensors with shapes: " (format "~a" shape1) " and " (format "~a" shape2)))
    (cond
      [(equal? shape1 shape2)
       (tensor shape1 (for/vector ([i (vector-length (tensor-data t1))])
                        (- (vector-ref (tensor-data t1) i)
                           (vector-ref (tensor-data t2) i))))]
      [else
       (error "Tensors must have the same shape for subtraction")])))

(define (tensor-multiply t1 t2)
  (let ([shape1 (tensor-shape t1)]
        [shape2 (tensor-shape t2)])
    (displayln (string-append "Multiplying tensors with shapes: " (format "~a" shape1) " and " (format "~a" shape2)))
    (cond
      [(and (= (length shape1) 2) (= (length shape2) 2) (= (cadr shape1) (car shape2)))
       (let* ([rows-a (car shape1)]
              [cols-a (cadr shape1)]
              [cols-b (cadr shape2)]
              [result (make-vector (* rows-a cols-b) 0.0)])
         (for ([i (in-range rows-a)])
           (for ([j (in-range cols-b)])
             (for ([k (in-range cols-a)])
               (vector-set! result (+ (* i cols-b) j)
                            (+ (vector-ref result (+ (* i cols-b) j))
                               (* (tensor-ref t1 i k) (tensor-ref t2 k j)))))))
         (tensor (list rows-a cols-b) result))]
      [(and (= (length shape1) 1) (= (length shape2) 2) (= (car shape1) (cadr shape2)))
       (let* ([rows-b (car shape2)]
              [cols-b (cadr shape2)]
              [result (make-vector rows-b 0.0)])
         (for ([i (in-range rows-b)])
           (for ([j (in-range cols-b)])
             (vector-set! result i
                          (+ (vector-ref result i)
                             (* (vector-ref (tensor-data t1) j) (tensor-ref t2 i j))))))
         (tensor (list rows-b) result))]
        [(and (= (length shape1) 2) (= (length shape2) 2) (equal? shape1 shape2))
       (tensor shape1 (vector-map * (tensor-data t1) (tensor-data t2)))]
      [else
       (error "Tensors must have compatible shapes for multiplication")])))
(define (tensor-ref t i j)
  (vector-ref (tensor-data t) (+ (* i (cadr (tensor-shape t))) j)))

(define (transpose t)
  (let* ([shape (tensor-shape t)]
         [rows (car shape)]
         [cols (cadr shape)]
         [data (tensor-data t)]
         [new-data (make-vector (apply * (reverse shape)) 0)])
    (for* ([i rows]
           [j cols])
      (vector-set! new-data (+ (* j rows) i) (vector-ref data (+ (* i cols) j))))
    (tensor (reverse shape) new-data)))

(define (scalar-multiply t scalar)
  (let ([data (tensor-data t)])
    (if (= (vector-length data) 1)
        (tensor (tensor-shape t) (vector (* (vector-ref data 0) scalar)))
        (tensor (tensor-shape t) (for/vector ([v data]) (* v scalar))))))

(define (dot-product t1 t2)
  (let ([data1 (tensor-data t1)]
        [data2 (tensor-data t2)])
    (if (not (= (length data1) (length data2)))
        (error "Tensors must have the same length for dot product")
        (apply + (map * data1 data2)))))
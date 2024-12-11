#lang racket

;expectes mnist files to be locally downloaded

(require "tensor.rkt")
(require "deep_learn_library.rkt")

;; MNIST Loading functions
(define (read-idx3-ubyte filename)
  (let* ([p (open-input-file filename #:mode 'binary)]
         [magic-number (integer-bytes->integer (read-bytes 4 p) #f #t)]
         [num-images (integer-bytes->integer (read-bytes 4 p) #f #t)]
         [num-rows (integer-bytes->integer (read-bytes 4 p) #f #t)]
         [num-cols (integer-bytes->integer (read-bytes 4 p) #f #t)]
         [data (make-vector (* num-images num-rows num-cols) 0)])
    (for ([i (in-range (vector-length data))])
      (vector-set! data i (read-byte p)))
    (close-input-port p)
    (t:create (list num-images (* num-rows num-cols)) data)))

(define (read-idx1-ubyte filename)
  (let* ([p (open-input-file filename #:mode 'binary)]
         [magic-number (integer-bytes->integer (read-bytes 4 p) #f #t)]
         [num-items (integer-bytes->integer (read-bytes 4 p) #f #t)]
         [data (make-vector num-items 0)])
    (for ([i (in-range num-items)])
      (vector-set! data i (read-byte p)))
    (close-input-port p)
    (t:create (list num-items 1) data)))

;; Load MNIST data
(define (load-mnist-data type)
  (let* ([base-path ""] ;; add path to mnist folder
         [images-file (string-append base-path 
                                     (if (equal? type "train")
                                         "train-images.idx3-ubyte"
                                         "t10k-images.idx3-ubyte"))]
         [labels-file (string-append base-path 
                                     (if (equal? type "train")
                                         "train-labels.idx1-ubyte"
                                         "t10k-labels.idx1-ubyte"))])
    (printf "Looking for images-file at: ~a~n" images-file)
    (printf "Looking for labels-file at: ~a~n" labels-file)

    ;; Error checking
    (unless (file-exists? images-file)
      (error 'load-mnist-data 
             (format "MNIST images file not found: ~a" images-file)))
    (unless (file-exists? labels-file)
      (error 'load-mnist-data 
             (format "MNIST labels file not found: ~a" labels-file)))
    (values (read-idx3-ubyte images-file)
            (read-idx1-ubyte labels-file))))

;; Normalize the data
(define (normalize X)
  (t:scale X (/ 1.0 255.0)))

;; One-hot encode labels
(define (one-hot y num-classes)
  (let* ([num-samples (car (t:shape y))]
         [encoded (make-vector (* num-samples num-classes) 0.0)])
    (for ([i (in-range num-samples)])
      (vector-set! encoded (+ (* i num-classes) (vector-ref (t:data y) i)) 1.0))
    (t:create (list num-samples num-classes) encoded)))

;; Get a batch of data given indices
(define (get-batch X indices)
  (let* ([batch-size (length indices)]
         [feature-size (cadr (t:shape X))]
         [batch-data (make-vector (* batch-size feature-size) 0.0)])
    (for ([i (range batch-size)]
          [idx indices])
      (for ([j (range feature-size)])
        (vector-set! batch-data 
                     (+ (* i feature-size) j)
                     (vector-ref (t:data X) (+ (* idx feature-size) j)))))
    (t:create (list batch-size feature-size) batch-data)))

;; Softmax function
(define (softmax z)
  (let* ([shape (t:shape z)]
         [data (t:data z)]
         [max-vals (for/vector ([i (in-range (car shape))])
                    (apply max (for/list ([j (in-range (cadr shape))])
                               (vector-ref data (+ (* i (cadr shape)) j)))))]
         [exp-vals (for/vector ([i (in-range (vector-length data))])
                    (exp (- (vector-ref data i) 
                           (vector-ref max-vals (quotient i (cadr shape))))))]
         [sum-vals (for/vector ([i (in-range (car shape))])
                    (for/sum ([j (in-range (cadr shape))])
                      (vector-ref exp-vals (+ (* i (cadr shape)) j))))])
    (t:create shape
              (for/vector ([i (in-range (vector-length exp-vals))])
                (/ (vector-ref exp-vals i)
                   (vector-ref sum-vals (quotient i (cadr shape))))))))

;; Forward pass
(define (broadcast-bias b batch-size)
  (let* ([bias-shape (t:shape b)]
         [num-classes (cadr bias-shape)]
         [expanded-data (make-vector (* batch-size num-classes) 0.0)])
    (for ([i (in-range batch-size)])
      (for ([j (in-range num-classes)])
        (vector-set! expanded-data (+ (* i num-classes) j)
                     (vector-ref (t:data b) j))))
    (t:create (list batch-size num-classes) expanded-data)))

(define (forward X w b)
  (let* ([z (t:add (t:mul X w) (broadcast-bias b (car (t:shape X))))]) ; Broadcast b
    (softmax z)))

;; Cross-entropy loss
(define (cross-entropy y-pred y-true)
  (let* ([m (car (t:shape y-true))]
         [epsilon 1e-15]
         [y-pred-clipped (t:create (t:shape y-pred)
                                 (for/list ([p (vector->list (t:data y-pred))])
                                   (max (min p (- 1.0 epsilon)) epsilon)))]
         [loss-vec (for/list ([i (in-range (vector-length (t:data y-true)))])
                    (* (vector-ref (t:data y-true) i)
                       (log (vector-ref (t:data y-pred-clipped) i))))])
    (- (/ (apply + loss-vec) m))))

;; Initialize parameters
(define input-size 784)  ; 28x28 pixels
(define num-classes 10)  ; digits 0-9
(define weights (t:random (list input-size num-classes) 0.01))  ; (input-size, num-classes)
(define bias (t:random (list 1 num-classes) 0.01))              ; (1, num-classes)

;; Training hyperparameters
(define learning-rate 0.1)
(define epochs 20)
(define batch-size 128)

;; Load and initialize MNIST data
(printf "Loading MNIST data...~n")
(define-values (X-train y-train) (load-mnist-data "train")) ; Load training data
(define-values (X-test y-test) (load-mnist-data "test"))    ; Load test data

(printf "Normalizing data...~n")
(set! X-train (normalize X-train)) ; Normalize X-train
(set! X-test (normalize X-test))   ; Normalize X-test

(printf "One-hot encoding labels...~n")
(set! y-train (one-hot y-train 10)) ; Convert y-train to one-hot
(set! y-test (one-hot y-test 10))   ; Convert y-test to one-hot

;; Training loop
(define (train-batch X-batch y-batch)
  (let* ([y-pred (forward X-batch weights bias)]
         [loss (cross-entropy y-pred y-batch)]
         [error (t:sub y-pred y-batch)]
         [gradient-w (t:scale (t:mul (t:transpose X-batch) error) 
                            (/ 1.0 (car (t:shape X-batch))))]
         [gradient-b (t:create (t:shape bias)
                             (for/list ([j (in-range num-classes)])
                               (/ (for/sum ([i (in-range (car (t:shape error)))])
                                    (vector-ref (t:data error) (+ (* i num-classes) j)))
                                  (car (t:shape error)))))])
    ;; Update parameters
    (set! weights (t:sub weights (t:scale gradient-w learning-rate)))
    (set! bias (t:sub bias (t:scale gradient-b learning-rate)))
    loss))

(define (get-test-accuracy X y)
  (let* ([predictions (forward X weights bias)]
         [num-samples (car (t:shape X))]
         [num-classes (cadr (t:shape y))]
         [pred-data (t:data predictions)]
         [true-data (t:data y)]
         [correct-count 0])
    
    ; Count correct predictions
    (for ([i (range num-samples)])
      (let* ([start-idx (* i num-classes)]
             ; Find predicted class (max probability index)
             [pred-vals (for/list ([j (range num-classes)])
                         (vector-ref pred-data (+ start-idx j)))]
             [pred-class (argmax (lambda (j) (list-ref pred-vals j))
                               (range num-classes))]
             ; Find true class 
             [true-class (for/first ([j (range num-classes)]
                                   #:when (= 1.0 (vector-ref true-data (+ start-idx j))))
                          j)])
        ; Increment counter if prediction matches truth
        (when (= pred-class true-class)
          (set! correct-count (add1 correct-count)))))
    
    ; Return accuracy as percentage
    (exact->inexact (* 100.0 (/ correct-count num-samples)))))

;; Main training loop
(printf "Starting training...~n")
(printf "Training on ~a samples, validating on ~a samples~n~n" 
        (car (t:shape X-train)) 
        (car (t:shape X-test)))

(define best-accuracy 0.0)

(for ([epoch (in-range epochs)])
  (printf "Epoch ~a/~a:~n" (add1 epoch) epochs)
  
  (let* ([indices (shuffle (range (car (t:shape X-train))))]
         [num-batches (quotient (length indices) batch-size)]
         [epoch-losses '()])

    ; Train on batches
    (for ([batch (in-range num-batches)])
      (let* ([batch-indices (take (drop indices (* batch batch-size)) batch-size)]
             [X-batch (get-batch X-train batch-indices)]
             [y-batch (get-batch y-train batch-indices)]
             [loss (train-batch X-batch y-batch)])
        
        (set! epoch-losses (cons loss epoch-losses))
        
        (when (= (modulo batch 50) 0)
          (printf "  Batch ~a/~a - Loss: ~a~n" 
                  batch 
                  num-batches 
                  (real->decimal-string loss 4)))))
    
    ; Epoch evaluation
    (let* ([avg-loss (/ (apply + epoch-losses) (length epoch-losses))]
           [test-accuracy (get-test-accuracy X-test y-test)])
      
      (when (> test-accuracy best-accuracy)
        (set! best-accuracy test-accuracy))
      
      (printf "~nEpoch Summary:~n")
      (printf "  Average Loss: ~a~n" (real->decimal-string avg-loss 4))
      (printf "  Test Accuracy: ~a%~n" (real->decimal-string test-accuracy 2))
      (printf "  Best Accuracy So Far: ~a%~n~n" (real->decimal-string best-accuracy 2)))))

(printf "Training Complete!~n")
(printf "Final Test Accuracy: ~a%~n" 
        (real->decimal-string (get-test-accuracy X-test y-test) 2))

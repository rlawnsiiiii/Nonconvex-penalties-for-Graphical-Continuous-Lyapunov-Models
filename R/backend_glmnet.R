#!/usr/bin/env Rscript
#
# glmnet backend for the Direct Lyapunov Lasso -- Dettling's own choice.
#
# This is a faithful transcription of `lassoB()` from
#   https://github.com/gherardovarando/gclm_experiments  (functions/util.R)
# which is how Varando & Hansen (2020) -- and, per its Appendix A, Dettling et
# al. (2024) -- actually fit the estimator: glmnet on the vectorized Lyapunov
# design matrix A(Sigma), with the diagonal of M left unpenalized.
#
# Usage:  Rscript R/backend_glmnet.R <input.json> <output.json>
#
# Input JSON:  { "Sigma": [[...]], "C": [[...]], "lambda_glmnet": [...] }
#   `lambda_glmnet` is on glmnet's own scale (see S1_reproduction.md Sec. 7.4).
# Output JSON: { "lambda_glmnet": [...], "beta": [[...]], "A": [[...]] }
#   `beta[[i]]` is vec(M_hat) (column-major) for lambda_glmnet[i].  `A`, the
#   design matrix, is returned only when "return_design": true -- it is p^2 x p^2
#   and must not be shipped back on production calls.

suppressPackageStartupMessages({
  library(glmnet)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
input <- fromJSON(args[1], simplifyMatrix = TRUE)

Sigma <- as.matrix(input$Sigma)
C <- as.matrix(input$C)
p <- nrow(Sigma)

## --- design matrix, exactly as in util.R::lassoB -------------------------
MM <- matrix(nrow = p, ncol = p, 1:(p^2))
TT <- diag(p^2)[c(t(MM)), ]                      # commutation matrix K(p,p)
AA <- Sigma %x% diag(p) + ((diag(p) %x% Sigma) %*% TT)

lambda <- as.numeric(input$lambda_glmnet)

## glmnet truncates a user-supplied lambda path by default: it stops early once
## the fractional deviance change falls below `fdev` or the deviance explained
## exceeds `devmax`, and caps the active set at `dfmax`/`pmax`.  For a
## reproduction we need every requested lambda fitted, so disable all of it.
##
## See the threshold ladder below for the second problem: glmnet's coordinate
## descent does not converge at tight `thresh` on this design, and then returns
## a truncated lambda sequence rather than an error.
old_ctrl <- glmnet.control()
on.exit(do.call(glmnet.control, old_ctrl), add = TRUE)
glmnet.control(fdev = 0, devmax = 1)

## Accuracy/robustness ladder.  Tighter `thresh` gives a more accurate fit, but
## on this rank-deficient design glmnet's coordinate descent stops converging
## near the dense end of the path and silently returns a TRUNCATED lambda
## sequence.  So: try tight first, loosen only as far as needed to fit every
## requested lambda, and report which threshold was actually used.
thresh_ladder <- if (is.null(input$thresh)) c(1e-14, 1e-12, 1e-10, 1e-8, 1e-7) else input$thresh

## glmnet >= 5.0 wants the convergence threshold via `control`; older versions
## take `thresh` directly. Pick whichever this installation supports.
use_control <- "control" %in% names(formals(glmnet))

fit <- NULL
thresh_used <- NA_real_
for (th in thresh_ladder) {
  call_args <- list(AA,
                    y = -c(C),
                    intercept = FALSE,
                    standardize = FALSE,
                    lambda = lambda,
                    dfmax = p^2 + 1,
                    pmax = p^2 + 1,
                    penalty.factor = 1 - diag(p))
  call_args[[if (use_control) "control" else "thresh"]] <-
    if (use_control) list(thresh = th) else th
  candidate <- suppressWarnings(do.call(glmnet, call_args))
  if (length(candidate$lambda) == length(lambda)) {
    fit <- candidate
    thresh_used <- th
    break
  }
}

if (is.null(fit)) {
  stop(sprintf(paste("glmnet could not fit all %d lambdas at any threshold in",
                     "[%s]; its coordinate descent does not converge on this",
                     "problem."),
               length(lambda), paste(thresh_ladder, collapse = ", ")))
}

beta <- as.matrix(fit$beta)

result <- list(lambda_glmnet = as.numeric(fit$lambda),
               beta = t(beta),           # row i <-> lambda i
               thresh_used = thresh_used,
               p = p)
if (isTRUE(input$return_design)) result$A <- AA

write_json(result, args[2], digits = 16, auto_unbox = TRUE)

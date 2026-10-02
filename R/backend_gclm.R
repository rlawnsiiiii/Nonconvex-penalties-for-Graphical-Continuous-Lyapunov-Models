#!/usr/bin/env Rscript
#
# gclm backend: Varando & Hansen's Algorithm 1 (proximal gradient) for the
# l1-penalised log-likelihood or Frobenius loss on the implied covariance,
# via gclm::gclm() (CRAN, v0.0.1).  Used as the reference the Python solver
# in src/gclm/solvers/covariance.py is validated against (tests/test_covloss.py).
#
# gclm() minimises   L(Sigma(B, C)) + lambda * sum_{i != j} |B_ij|
# with C diagonal and FIXED when lambdac < 0 (its Fortran updates C only for
# lambdac >= 0).  L is
#   loss = "loglik":    log det Sigma + tr(Sigma^{-1} Sigma_hat)
#   loss = "frobenius": 0.5 * ||Sigma - Sigma_hat||_F^2
# -- the same scale as src/gclm/objective/covariance.py, so lambda passes through unchanged.
# Its stopping rule is an objective decrease of at most `eps`, absolute or
# relative, or `maxIter` iterations; with eps = 0 it runs until no decrease is
# representable, which is what the validation tests ask for.
#
# Usage: Rscript R/backend_gclm.R <input.json> <output.json>
# Input:  { "Sigma": [[..]], "C": [..] (diagonal), "B0": [[..]] (start),
#           "lambda": [..], "loss": "loglik"|"frobenius",
#           "eps": <num>, "maxIter": <int> }
#   Every lambda is solved from the same start B0 (no warm starts), so that
#   each fit is a self-contained reference point.
# Output: { "lambda": [..], "B": [[[..]]], "loss": [..], "iter": [..] }
#   "loss" is the smooth part L at the solution (gclm's `loss` return value).

suppressPackageStartupMessages({
  library(gclm)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
input <- fromJSON(args[1], simplifyMatrix = TRUE)

Sigma   <- as.matrix(input$Sigma)
Cdiag   <- as.numeric(input$C)
B0      <- as.matrix(input$B0)
lambdas <- as.numeric(input$lambda)
loss    <- if (is.null(input$loss)) "loglik" else input$loss
eps     <- if (is.null(input$eps)) 0 else as.numeric(input$eps)
maxIter <- if (is.null(input$maxIter)) 100000L else as.integer(input$maxIter)

Bs <- vector("list", length(lambdas))
ls <- numeric(length(lambdas))
it <- integer(length(lambdas))
for (i in seq_along(lambdas)) {
  fit <- gclm(Sigma, B = B0, C = Cdiag, C0 = Cdiag, loss = loss,
              eps = eps, alpha = 0.5, maxIter = maxIter,
              lambda = lambdas[i], lambdac = -1, job = 0)
  Bs[[i]] <- fit$B
  ls[i]   <- fit$loss
  it[i]   <- fit$iter
}

write_json(list(lambda = lambdas, B = Bs, loss = ls, iter = it),
           args[2], digits = 16, auto_unbox = TRUE)

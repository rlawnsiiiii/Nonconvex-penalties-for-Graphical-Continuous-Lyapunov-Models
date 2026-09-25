#!/usr/bin/env Rscript
#
# Reference implementation for S1b (MCP / SCAD) -- and a second opinion on S1's
# lasso -- via ncvreg::ncvfit().
#
# ncvfit() is the low-level fitter: unlike ncvreg(), it does NOT standardise the
# design and does NOT fit an intercept, which is what our problem requires
# (y = -vec(C) has no constant term, and the columns of A(Sigma) are on
# meaningful, comparable scales that must not be rescaled).
#
# Usage: Rscript R/backend_ncvreg.R <input.json> <output.json>
# Input:  { "Sigma": [[..]], "C": [[..]], "lambda_ncv": [..],
#           "penalty": "lasso"|"MCP"|"SCAD", "gamma": <num> }
#   lambda_ncv is on ncvfit's own scale: (1/(2n))||y-Xb||^2 + lambda*pf_j*pen(b_j)
#   with n = nrow(X) = p^2.
# Output: { "lambda_ncv": [..], "beta": [[..]], "iter": [..] }

suppressPackageStartupMessages({
  library(ncvreg)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
input <- fromJSON(args[1], simplifyMatrix = TRUE)

Sigma <- as.matrix(input$Sigma)
C     <- as.matrix(input$C)
p     <- nrow(Sigma)

## design matrix A(Sigma), same construction as R/reference_lasso.R
MM <- matrix(nrow = p, ncol = p, 1:(p^2))
TT <- diag(p^2)[c(t(MM)), ]
AA <- Sigma %x% diag(p) + ((diag(p) %x% Sigma) %*% TT)
y  <- -c(C)

penalty <- if (is.null(input$penalty)) "lasso" else input$penalty
gamma   <- if (is.null(input$gamma)) switch(penalty, SCAD = 3.7, 3) else input$gamma
pf      <- c(1 - diag(p))            # diagonal of M unpenalised
lambdas <- as.numeric(input$lambda_ncv)

xtx <- apply(AA, 2, crossprod) / nrow(AA)

betas <- matrix(0, nrow = length(lambdas), ncol = p^2)
iters <- integer(length(lambdas))
init  <- rep(0, p^2)
## walk the path from large to small lambda, warm-starting (ncvfit is a
## single-lambda fitter; pathwise continuation is the caller's job)
ord <- order(lambdas, decreasing = TRUE)
for (i in ord) {
  fit <- ncvfit(AA, y, init = init, xtx = xtx,
                penalty = penalty, gamma = gamma, lambda = lambdas[i],
                penalty.factor = pf, eps = 1e-12, max.iter = 100000,
                warn = FALSE)
  betas[i, ] <- fit$beta
  iters[i]   <- fit$iter
  init       <- fit$beta
}

write_json(list(lambda_ncv = lambdas, beta = betas, iter = iters,
                penalty = penalty, gamma = gamma),
           args[2], digits = 16, auto_unbox = TRUE)

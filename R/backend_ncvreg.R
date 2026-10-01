#!/usr/bin/env Rscript
#
# ncvreg backend: lasso, MCP and SCAD for the Direct Lyapunov estimator, via
# ncvreg::ncvfit() (Breheny & Huang 2011).
#
# ncvfit() is the low-level fitter.  Unlike ncvreg() it applies no
# standardisation and adds no intercept -- both wrong here: y = -vec(C) has no
# constant term, and the columns of A(Sigma) carry the problem's own scaling.
#
# SCALING.  ncvfit minimises   (1/(2n)) ||y - X b||^2 + sum_j P_{lambda*pf_j, gamma}(b_j)
# with n = nrow(X).  We want Dettling's paper scale
#                              (1/2)    ||y - X b||^2 + sum_j P_{lambda*pf_j, gamma}(b_j).
# Passing sqrt(n) * X and sqrt(n) * y makes the two identical, so lambda and
# gamma go through UNCHANGED for lasso, MCP and SCAD alike.  The alternative --
# keep X unscaled and convert the parameters -- works for the lasso and for MCP,
# but not for SCAD: n * SCAD_{lambda', gamma'} is not a SCAD penalty for any
# (lambda, gamma), because SCAD's kinks sit at lambda itself.  See
# docs/NONCONVEX.md Section 2 and R/ENCODING.md Section 4.
#
# Usage: Rscript R/backend_ncvreg.R <input.json> <output.json>
# Input:  { "Sigma": [[..]], "C": [[..]], "lambda": [..],
#           "penalty": "lasso"|"MCP"|"SCAD", "gamma": <num, optional> }
#   lambda and gamma are on the paper scale -- the same numbers every other
#   backend in this repository takes.
# Output: { "lambda": [..], "beta": [[..]], "iter": [..], "penalty", "gamma" }

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

## design matrix A(Sigma) -- same construction as backend_glmnet.R
MM <- matrix(nrow = p, ncol = p, 1:(p^2))
TT <- diag(p^2)[c(t(MM)), ]
AA <- Sigma %x% diag(p) + ((diag(p) %x% Sigma) %*% TT)
y  <- -c(C)

## paper scale: (1/(2n)) ||sqrt(n) y - sqrt(n) X b||^2 = (1/2) ||y - X b||^2
n  <- nrow(AA)
Xs <- sqrt(n) * AA
ys <- sqrt(n) * y

penalty <- if (is.null(input$penalty)) "lasso" else input$penalty
gamma   <- if (is.null(input$gamma)) switch(penalty, SCAD = 3.7, 3) else input$gamma
pf      <- c(1 - diag(p))            # diagonal of M unpenalised
lambdas <- as.numeric(input$lambda)

xtx <- apply(Xs, 2, crossprod) / n    # = squared column norms of the unscaled A

betas <- matrix(0, nrow = length(lambdas), ncol = p^2)
iters <- integer(length(lambdas))
init  <- rep(0, p^2)
## Walk the path from large to small lambda, warm-starting each fit from the
## previous one.  ncvfit is a single-lambda fitter and leaves continuation to
## the caller; for MCP/SCAD its help page stresses that "initial values are
## very important in determining which local solution an algorithm converges to".
ord <- order(lambdas, decreasing = TRUE)
for (i in ord) {
  fit <- ncvfit(Xs, ys, init = init, xtx = xtx,
                penalty = penalty, gamma = gamma, lambda = lambdas[i],
                penalty.factor = pf, eps = 1e-12, max.iter = 100000,
                warn = FALSE)
  betas[i, ] <- fit$beta
  iters[i]   <- fit$iter
  init       <- fit$beta
}

write_json(list(lambda = lambdas, beta = betas, iter = iters,
                penalty = penalty, gamma = gamma),
           args[2], digits = 16, auto_unbox = TRUE)

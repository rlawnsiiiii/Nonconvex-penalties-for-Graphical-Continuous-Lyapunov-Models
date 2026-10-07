#!/usr/bin/env Rscript
#
# Paths of Varando & Hansen's penalised log-likelihood estimator (gclm::gclm, their Algorithm 1)
# on one correlation matrix, for several treatments of the volatility matrix C.  Called by
# estimate_c_check.py; not part of the library.
#
# Usage: Rscript estimate_c_check.R <input.json> <output.json>
# Input:  { "R": [[..]] correlation matrix, "lambdas": [..] increasing,
#           "variants": [ { "name": .., "C": [..] start (diagonal), "C0": [..] ridge target,
#                           "lambdac": <num, negative = C fixed>, "direction": "up"|"down" } ],
#           "eps": <num>, "maxIter": <int> }
# "up":   start from B0 = -0.5 * C * R^{-1} (exact fit) and walk the lambdas in increasing order,
#         warm-starting B and C (the protocol of Varando & Hansen's simulations).
# "down": start from B = -0.5 * diag(C) and walk in decreasing order (sparse -> dense).
# Output: per variant the list of B (in the order of "lambdas"), of C, iterations and loss.

suppressPackageStartupMessages({
  library(gclm)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
input <- fromJSON(args[1], simplifyMatrix = TRUE, simplifyDataFrame = FALSE)

R       <- as.matrix(input$R)
lambdas <- as.numeric(input$lambdas)
eps     <- as.numeric(input$eps)
maxIter <- as.integer(input$maxIter)
p       <- ncol(R)
Rinv    <- solve(R)

out <- list()
for (v in input$variants) {
  Cd <- as.numeric(v$C)
  C0 <- as.numeric(v$C0)
  up <- identical(v$direction, "up")
  B  <- if (up) -0.5 * diag(Cd) %*% Rinv else -0.5 * diag(Cd)
  ord <- if (up) seq_along(lambdas) else rev(seq_along(lambdas))
  Bs <- vector("list", length(lambdas)); Cs <- vector("list", length(lambdas))
  it <- integer(length(lambdas)); ls <- numeric(length(lambdas))
  for (i in ord) {
    fit <- gclm(R, B = B, C = Cd, C0 = C0, loss = "loglik", eps = eps, alpha = 0.5,
                maxIter = maxIter, lambda = lambdas[i], lambdac = as.numeric(v$lambdac), job = 0)
    B <- fit$B; Cd <- fit$C
    Bs[[i]] <- B; Cs[[i]] <- Cd; it[i] <- fit$iter; ls[i] <- fit$loss
  }
  out[[v$name]] <- list(B = Bs, C = Cs, iter = it, loss = ls)
}
write_json(out, args[2], digits = 16, auto_unbox = TRUE)

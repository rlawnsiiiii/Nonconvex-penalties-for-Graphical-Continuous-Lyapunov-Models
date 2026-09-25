#!/usr/bin/env Rscript
#
# Reference metric computations, transcribed from
#   https://github.com/gherardovarando/gclm_experiments (functions/util.R)
# functions FPR(), TPR(), AUROC(), AUCPR() and evaluatePathB().
#
# Usage: Rscript R/reference_metrics.R <input.json> <output.json>
# Input:  { "estimates": [ [[...]], ... ],  "M_star": [[...]] }
#         estimates ordered by INCREASING lambda.

suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
input <- fromJSON(args[1], simplifyMatrix = TRUE)

B <- as.matrix(input$M_star)
p <- nrow(B)
ests <- input$estimates                    # 3-d array: [index, row, col]
n <- dim(ests)[1]
ix <- lower.tri(B) | upper.tri(B)          # off-diagonal only

FPR <- function(x, y) if (sum(y == 0) == 0) 1 else sum(x != 0 & y == 0) / sum(y == 0)
TPR <- function(x, y) if (sum(y != 0) == 0) 1 else sum(x != 0 & y != 0) / sum(y != 0)

conf <- t(sapply(seq_len(n), function(i) {
  Bh <- matrix(ests[i, , ], nrow = p)
  tp <- sum(Bh[ix] != 0 & B[ix] != 0); fp <- sum(Bh[ix] != 0 & B[ix] == 0)
  fn <- sum(Bh[ix] == 0 & B[ix] != 0); tn <- sum(Bh[ix] == 0 & B[ix] == 0)
  c(tp = tp, fp = fp, fn = fn, tn = tn)
}))
conf <- as.data.frame(conf)
conf$tpr <- conf$tp / (conf$tp + conf$fn); conf$tpr[is.nan(conf$tpr)] <- 1
conf$fpr <- conf$fp / (conf$fp + conf$tn); conf$fpr[is.nan(conf$fpr)] <- 1
conf$acc <- (conf$tp + conf$tn) / (conf$tp + conf$tn + conf$fp + conf$fn)
conf$precision <- conf$tp / (conf$tp + conf$fp); conf$precision[is.nan(conf$precision)] <- 1
conf$recall <- conf$tpr
conf$f1 <- 2 * conf$tp / (2 * conf$tp + conf$fp + conf$fn); conf$f1[is.nan(conf$f1)] <- 0

## AUROC: reverse the path, anchor with (0,0) and (1,1), trapezoid.
roc <- cbind(FPR = conf$fpr, TPR = conf$tpr)
roc <- roc[nrow(roc):1, , drop = FALSE]
roc <- rbind(c(0, 0), roc, c(1, 1))
auc <- 0
for (i in 1:(nrow(roc) - 1)) {
  auc <- auc + (roc[i + 1, 2] + roc[i, 2]) * (roc[i + 1, 1] - roc[i, 1]) / 2
}

## AUCPR, as in util.R
aupr <- 0
for (i in 1:(n - 1)) {
  aupr <- aupr + (conf$precision[i + 1] + conf$precision[i]) *
    (conf$recall[i] - conf$recall[i + 1]) / 2
}
aupr <- aupr + (1 + conf$precision[n]) * (conf$recall[n] - 0) / 2

write_json(list(tpr = conf$tpr, fpr = conf$fpr, acc = conf$acc,
                f1 = conf$f1, precision = conf$precision,
                max_acc = max(conf$acc), max_f1 = max(conf$f1),
                auc = auc, aupr = aupr),
           args[2], digits = 16, auto_unbox = TRUE)

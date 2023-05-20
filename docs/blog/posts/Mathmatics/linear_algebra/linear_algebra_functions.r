# Trace
tr<-function(mat){
  return(sum(diag(mat)))
}
#determinant
## Cofactors
cofactor<-function(mat,i,j){
  mat_sub<-mat[-i,-j]
  return((-1)^(i+j)*det(mat_sub))
}
cofactor_matrix<-function(mat){
  n<-nrow(mat)
  if(n!=ncol(mat)){
    stop('the matrix is not a square matrix')
  }
  cofactors<-matrix(nrow=n,ncol=n)
  for (i in 1:n){
    for (j in 1:n){
      cofactors[i,j]<-cofactor(mat,i,j)
    }
  }
  return(cofactors)
}

cofactor_matrix2<-function(mat){
  n<-nrow(mat)
  if(n!=ncol(mat)){
    stop('the matrix is not a square matrix')
  }
  coordinate_set<-expand.grid(1:n,1:n)
  cofactors<-mapply(cofactor,
  i=coordinate_set[,1],
  j=coordinate_set[,2],
  MoreArgs=list(mat=A))
  return(
  matrix(cofactors,ncol=n)
  )
}


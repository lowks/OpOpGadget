#include <stdlib.h>
#include <math.h>
#include <stdio.h>

int main(void)
{
   FILE *file = fopen("dp.txt", "r");
   if ( file != NULL )
   {
      size_t i, count, size = 512;
      double *x = malloc(size * sizeof(*x));
      double *y = malloc(size * sizeof(*y));
      if ( x != NULL && y != NULL  )
      {
         /* Read all three columns. */
         for ( count = 0; count < size; ++count )
         {
            if ( fscanf (file ,"%lf%lf", &x[count], &y[count] ))
            {
               break;
            }
         }
         /* Use only the first two columns. */
         for ( i = 0; i < count; ++i )
         {
            printf("x = %f, y = %f\n", x[i], y[i]);
         }
         free(x);
         free(y);

      }
   }
   return 0;
}
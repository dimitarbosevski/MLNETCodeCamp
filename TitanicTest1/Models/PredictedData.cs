using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TitanicTest1.Models
{
    public class PredictedData
    {
        [ColumnName("PredictedLabel")]
        public bool IsSurvived;


        [ColumnName("Score")]
        public float Score;
    }
}

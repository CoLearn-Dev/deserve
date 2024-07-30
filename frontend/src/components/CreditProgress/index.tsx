import { Progress } from 'antd-v5';

import React from 'react';
import { useEffect, useState } from 'react';

const CreditProgress: React.FC = ({ credit, width, size, type }) => {
  var credit = (credit / 1000000).toFixed(3);
  return (
    <Progress
      percent={Math.min(Math.log(credit + 1) / Math.log(1.1), 100)}
      format={() =>
        type == 'line' ? (
          `${credit} million fleece tokens`
        ) : (
          <div>
            <b style={{ color: '#FDB515' }}>{`${credit} `}</b> <br />
            <text style={{ fontSize: 16 }}>million fleece tokens</text>
          </div>
        )
      }
      type={type ? type : 'line'}
      size={size}
      style={{ width: width }}
      status="active"
      strokeColor={{ '0%': '#fd5315', '100%': '#FDB515' }}
    />
  );
};

export default CreditProgress;

// k6 load test for the inference endpoint.
//   k6 run -e API_URL=http://localhost:8001 -e SERIES="FOODS_1_005:TX_3,FOODS_2_159:TX_3" tests/load/infer.js
// Run it at 1 / 3 / 5 pods (kubectl scale or let the HPA do it) and write RPS + p95 into docs/results.md.
// Chaos-lite: while this runs, `kubectl delete pod -l app=demand-forecaster --wait=false` for ONE pod:
// the http_req_failed threshold shows whether any request failed.
import http from 'k6/http';
import { check } from 'k6';

const API = __ENV.API_URL || 'http://localhost:8001';
const SERIES = (__ENV.SERIES || 'FOODS_1_005:TX_3').split(',').map((s) => s.split(':'));

export const options = {
  scenarios: {
    steady: {
      executor: 'constant-arrival-rate',
      rate: Number(__ENV.RATE || 50), // requests per second
      timeUnit: '1s',
      duration: __ENV.DURATION || '60s',
      preAllocatedVUs: 50,
      maxVUs: 200,
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.001'],
    http_req_duration: ['p(95)<100'], // the same 100 ms budget the promotion gate uses
  },
};

export default function () {
  const [item, store] = SERIES[Math.floor(Math.random() * SERIES.length)];
  const res = http.post(
    `${API}/v2/models/demand-forecaster/infer`,
    JSON.stringify({ item_id: item, store_id: store, horizon: 28 }),
    { headers: { 'Content-Type': 'application/json' } },
  );
  check(res, {
    'status 200': (r) => r.status === 200,
    '28 days returned': (r) => r.status === 200 && r.json('forecasts').length === 28,
  });
}

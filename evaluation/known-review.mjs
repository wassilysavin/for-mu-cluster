import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {canonical} from './schema.mjs';
import {gradeTurn} from './evaluation.mjs';

const sha=value=>createHash('sha256').update(value).digest('hex');
const unique=batches=>[...new Map(batches.map(batch=>[canonical(batch),batch])).values()];

export async function applyKnownReview(registry, row, record, policy) {
  const contextKey=`${record.case_id}:${record.before_visible_history_sha256}`;
  const meaning=policy.contexts[contextKey];
  const exchange=record.after.exchanges.at(-1);
  const verdict=policy.verdicts[`${contextKey}:${sha(exchange.raw_output??'')}`];
  const amendments=policy.amendments.filter(rule=>rule.case_ids.includes(row.case_id));
  const extra=amendments.flatMap(rule=>rule.additional_allowed_batches);
  const sharing=policy.sharing_amendment;
  if(sharing.affected_case_ids.includes(row.case_id)){
    const original=record.before.exchanges.find(value=>value.id==='e1');
    const failed=row.case_id.endsWith('-t3')&&original&&original.outcome!=='committed';
    extra.push(...(failed?sharing.allowed_batches_for_t3_when_actual_original_e1_failed:
      sharing.allowed_batches_when_original_not_failed));
  }
  if(!meaning&&!extra.length)return record;
  if(meaning)assert.equal(meaning.question_sha256,sha(row.question));
  const allowed=unique([...(meaning?.allowed_batches??row.allowed_model_call_batches),...extra]);
  const gold={...row,author:'known-development-author',history_policy:'reviewed_actual_candidate',
    allowed_model_call_batches:allowed,review:{status:'approved',reviewer:policy.reviewer,
      registry_sha256:policy.registry_sha256,question_sha256:sha(row.question),
      before_visible_history_sha256:record.before_visible_history_sha256}};
  let graded;
  try{graded=await gradeTurn(registry,record.before,record.after,gold);}
  catch(error){if(error.code!=='unapproved_gold')throw error;return record;}
  if(graded.success&&verdict?.critical_status==='detected')throw new Error('Known review contradicts semantic acceptance');
  const result={...graded,critical_status:graded.success?'none_observed':verdict?.critical_status??graded.critical_status};
  return {...record,result,known_review:{policy_version:policy.schema_version,
    known_context:!!meaning,known_output:!!verdict,amendments:amendments.map(rule=>rule.rule_id),
    reviewer:policy.reviewer,category:verdict?.category??null}};
}

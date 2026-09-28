import {test} from 'node:test';
import assert from 'node:assert/strict';
import {appendSample,format,currentAssessment,hasSessionCache,displayedRul} from './api.js';
test('history deduplicates ticks, clears on session reset and remains bounded',()=>{
 let h=[];for(let timestamp=0;timestamp<220;timestamp++)h=appendSample(h,{session_id:'a',timestamp});
 assert.equal(h.length,180);assert.equal(appendSample(h,h.at(-1)),h);
 assert.equal(appendSample(h,{session_id:'b',timestamp:1}).length,1);
 assert.equal(appendSample(h,{session_id:'a',timestamp:0}).length,1);
});
test('unavailable model outputs never become zero',()=>{assert.equal(format(null),'Unavailable');assert.equal(format(NaN),'Unavailable');assert.equal(format(0),'0.0');});
test('old control revisions cannot be shown as current predictions',()=>{
 const p={control_revision:4,telemetry:{control_pending:false,assessment_control_revision:4}};
 assert.equal(currentAssessment(p,5),false);assert.equal(currentAssessment(p,4),true);
 p.telemetry.assessment_control_revision=3;assert.equal(currentAssessment(p,3),false);
});
test('missing sessions never match an empty assessment cache',()=>{
 assert.equal(hasSessionCache(null,undefined),false);
 assert.equal(hasSessionCache(null,'session-a'),false);
 assert.equal(hasSessionCache({sessionId:'session-a'},'session-a'),true);
 assert.equal(hasSessionCache({sessionId:'session-a'},'session-b'),false);
});
test('normal-state display adds ten hours without changing the source estimate',()=>{
 const source={hours:12,interval_hours:[8,20]};
 assert.deepEqual(displayedRul(source,'normal'),{hours:22,interval:[8,20],uplift:10});
 assert.deepEqual(displayedRul(source,'abnormal'),{hours:12,interval:[8,20],uplift:0});
 assert.deepEqual(displayedRul({hours:null},'normal'),{hours:null,interval:null,uplift:0});
 assert.equal(source.hours,12);
});
